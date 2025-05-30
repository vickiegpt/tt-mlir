// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttmlir/Dialect/TT/IR/TT.h"
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Iterators.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::tt::ttir {
#define GEN_PASS_DEF_TTIROPTIMIZETENSORLAYOUT
#include "ttmlir/Dialect/TTIR/Transforms/Passes.h.inc"

static GridAttr getOptimalGrid(PatternRewriter &rewriter,
                               ArrayRef<int64_t> memrefShape,
                               ArrayRef<int64_t> deviceGridShape) {
  std::vector<int64_t> gridShape;
  for (size_t i = 0; i < memrefShape.size(); i++) {
    int64_t dim = memrefShape[i];
    for (size_t grid = deviceGridShape[i]; grid > 0; grid--) {
      if (dim % grid == 0) {
        gridShape.push_back(grid);
        break;
      }
    }
  }
  return rewriter.getAttr<GridAttr>(gridShape);
}

static RankedTensorType calculateOptimalLayoutForTensorType(
    PatternRewriter &rewriter, Value tensor,
    const SmallVector<int64_t> &workerGridShape) {
  RankedTensorType resultType = mlir::cast<RankedTensorType>(tensor.getType());
  auto resultEncoding =
      mlir::cast_if_present<MetalLayoutAttr>(resultType.getEncoding());
  assert(resultEncoding && "Tensor type must have a MetalLayoutAttr encoding");
  assert(resultEncoding.getGrid().getShape().size() ==
             resultEncoding.getMemref().getShape().size() &&
         "Grid rank must match memref rank.");

  SmallVector<int64_t> canonicalShape;
  canonicalShape.reserve(resultEncoding.getMemref().getShape().size());

  for (size_t i = 0; i < resultEncoding.getMemref().getShape().size(); i++) {
    canonicalShape.push_back(resultEncoding.getMemref().getShape()[i] *
                             resultEncoding.getGrid().getShape()[i]);
  }

  auto optimalOutputGrid =
      getOptimalGrid(rewriter, canonicalShape, workerGridShape);
  auto newResultEncoding = resultEncoding.withGrid(
      tensor.getContext(), resultType, optimalOutputGrid);

  return RankedTensorType::get(resultType.getShape(),
                               resultType.getElementType(), newResultEncoding);
}

static SmallVector<int64_t>
calculateOutputBlockFactors(ArrayRef<int64_t> outputShardShape,
                            unsigned dstRegisterSizeTiles) {
  // The output operand always corresponds to the compute grid and is therefore
  // the only shape we care about when it comes to constraining on the dst
  // register size. We reverse the output shape to give the priority to the
  // inner dimension and to ensure row major output.
  int64_t remainingBlockFactor = static_cast<int64_t>(dstRegisterSizeTiles);
  SmallVector<int64_t> outputBlockFactors;
  outputBlockFactors.reserve(outputShardShape.size());
  for (int64_t dim : llvm::reverse(outputShardShape)) {
    for (int64_t factor = remainingBlockFactor; factor > 0; factor--) {
      if (dim % factor == 0) {
        outputBlockFactors.push_back(dim / factor);
        // If the dimension is fully consumed by this factor, then we can
        // continue pulling factors from outer dimensions. Otherwise we must
        // snap it to 1 to enforce row major output.
        bool consumed = dim == factor;
        remainingBlockFactor = consumed ? remainingBlockFactor / factor : 1;
        assert(remainingBlockFactor > 0);
        break;
      }
    }
  }
  assert(outputBlockFactors.size() == outputShardShape.size());
  // We reversed on the way in, so reverse it back.
  return llvm::to_vector(llvm::reverse(outputBlockFactors));
}

static SmallVector<int64_t>
calculateOptimalBlockFactors(ArrayRef<AffineMap> indexingMaps,
                             ArrayRef<int64_t> outputShardShape,
                             unsigned dstRegisterSizeTiles) {
  assert(!indexingMaps.empty());
  MLIRContext *context = indexingMaps[0].getContext();

  SmallVector<int64_t> outputBlockFactors =
      calculateOutputBlockFactors(outputShardShape, dstRegisterSizeTiles);

  //
  // Concat all of the indexing maps together, matmul example:
  // (d0, d1, d2) -> (d0, d2)
  // (d0, d1, d2) -> (d2, d1)
  // (d0, d1, d2) -> (d0, d1)
  // Becomes:
  // (d0, d1, d2) -> (d0, d2, d2, d1, d0, d1)
  //
  // We reverse it so that output dimensions get priority for the inverse
  // permutation.
  //
  SmallVector<AffineMap> indexingMapsReversed =
      llvm::to_vector(llvm::reverse(indexingMaps));
  AffineMap concat = concatAffineMaps(indexingMapsReversed, context);

  //
  // Invert the permutation to get a map that we can use to get the buffer
  // factors. Above example becomes:
  // (d0, d1, d2, d3, d4, d5) -> (d0, d3, d1)
  //
  AffineMap inverse = inversePermutation(concat);

  //
  // Since we reversed above to give the output block factors priority in the
  // inverse affine map, we add those first. Then fill the rest of the dims with
  // 1s, these are free variables that don't depend on the sizing of dst. In the
  // future we might do something more intellegent with the free variables, or
  // enable downstream passes like allocation to adjust them based on memory
  // requirements. factors.
  //
  SmallVector<int64_t> flattenedBlockFactors(outputBlockFactors);
  flattenedBlockFactors.resize(inverse.getNumDims(), 1);

  // Eval the affine map to get the buffer factors.
  return inverse.compose(flattenedBlockFactors);
}

namespace {
struct TTIRGenericTensorLayoutRewriter : public OpRewritePattern<GenericOp> {
  TTIRGenericTensorLayoutRewriter(MLIRContext *context,
                                  SmallVector<int64_t> workerGridShape,
                                  unsigned dstRegisterSizeTiles)
      : OpRewritePattern<GenericOp>(context), workerGridShape(workerGridShape),
        dstRegisterSizeTiles(dstRegisterSizeTiles) {}

  LogicalResult matchAndRewrite(GenericOp op,
                                PatternRewriter &rewriter) const final {
    // Update output tensor type
    assert(op->getResults().size() == 1 &&
           "Only one result tensor is supported for now");
    auto newTensorType = calculateOptimalLayoutForTensorType(
        rewriter, op->getResult(0), workerGridShape);
    MetalLayoutAttr metalLayout =
        mlir::cast<MetalLayoutAttr>(newTensorType.getEncoding());
    SmallVector<int64_t> outputShardShape =
        metalLayout.getShardShape(/*convertTileToScalar=*/false);
    SmallVector<int64_t> blockFactors = calculateOptimalBlockFactors(
        op.getIndexingMapsValue(), outputShardShape, dstRegisterSizeTiles);
    bool blockFactorsChanged = blockFactors != op.getBlockFactorsValue();
    if (op.getGrid() == metalLayout.getGrid() && !blockFactorsChanged) {
      return failure();
    }

    Type originalType = op->getResult(0).getType();
    auto dpsOp = mlir::cast<DestinationStyleOpInterface>(op.getOperation());
    assert(dpsOp.getNumDpsInits() == 1 &&
           "Only one result tensor is supported for now");
    for (OpOperand &operand : op->getOpOperands()) {
      auto newOperandType = calculateOptimalLayoutForTensorType(
          rewriter, operand.get(), workerGridShape);
      if (operand.get().getType() != newOperandType || blockFactorsChanged) {
        Value view;
        MetalLayoutAttr viewMetalLayout;
        std::tie(view, viewMetalLayout) =
            blockedView(rewriter, op->getLoc(), operand.get(), newOperandType,
                        op.getIndexingMapsValue()[operand.getOperandNumber()],
                        blockFactors);
        rewriter.modifyOpInPlace(op, [&]() { operand.set(view); });

        if (dpsOp.isDpsInit(&operand)) {
          assert(newOperandType == newTensorType &&
                 "DPS init tensor must have the same type as the result");
          rewriter.modifyOpInPlace(
              op, [&]() { op->getResult(0).setType(view.getType()); });
        }

        for (auto &region : op->getRegions()) {
          assert(region.getBlocks().size() == 1 &&
                 "Only one block per region is supported.");
          Block &genericBlock = region.front();
          auto arg = genericBlock.getArgument(operand.getOperandNumber());
          rewriter.modifyOpInPlace(
              op, [&]() { arg.setType(viewMetalLayout.getMemref()); });
        }
      }
    }

    rewriter.modifyOpInPlace(op, [&]() {
      // Update generic grid (match worker cores to output grid)
      op.setGridAttr(
          mlir::cast<MetalLayoutAttr>(newTensorType.getEncoding()).getGrid());
      op.setBlockFactorsAttr(rewriter.getI64ArrayAttr(blockFactors));
    });

    rewriter.setInsertionPointAfter(op);
    auto emptyOp = rewriter.create<EmptyOp>(op->getLoc(), originalType);
    auto toLayoutOp = rewriter.create<ToLayoutOp>(
        op->getLoc(), op->getResult(0), emptyOp.getResult());
    rewriter.replaceAllUsesExcept(op->getResult(0), toLayoutOp.getResult(0),
                                  toLayoutOp);

    return success();
  }

  static std::pair<Value, MetalLayoutAttr>
  blockedView(PatternRewriter &rewriter, Location loc, Value tensor,
              RankedTensorType newOperandType, AffineMap indexingMap,
              ArrayRef<int64_t> blockFactors) {
    auto emptyOp = rewriter.create<EmptyOp>(loc, newOperandType);
    auto toLayoutOp =
        rewriter.create<ToLayoutOp>(loc, tensor, emptyOp.getResult());
    MetalLayoutAttr metalLayout =
        mlir::cast<MetalLayoutAttr>(newOperandType.getEncoding());
    SmallVector<int64_t> blockShape = indexingMap.compose(blockFactors);
    for (auto [i, dim] : llvm::enumerate(metalLayout.getGrid().getShape())) {
      blockShape[i] *= dim;
    }
    MetalLayoutAttr viewLayout = metalLayout.withGrid(blockShape);
    return std::make_pair(
        rewriter
            .create<ViewLayoutOp>(
                loc,
                RankedTensorType::get(newOperandType.getShape(),
                                      newOperandType.getElementType(),
                                      viewLayout),
                toLayoutOp.getResult(0))
            .getResult(),
        viewLayout);
  }

  SmallVector<int64_t> workerGridShape;
  unsigned dstRegisterSizeTiles;
};
} // namespace

namespace {
class TTIROptimizeTensorLayout
    : public impl::TTIROptimizeTensorLayoutBase<TTIROptimizeTensorLayout> {

  using impl::TTIROptimizeTensorLayoutBase<
      TTIROptimizeTensorLayout>::TTIROptimizeTensorLayoutBase;

  void runOnOperation() final {
    auto device = lookupDevice(getOperation());
    assert(device && "Device not found");
    auto systemDesc = getCurrentScopeSystemDesc(getOperation());
    auto chipIds = device.getChipIds();
    assert(chipIds.size() == 1);
    auto chipDesc = systemDesc.getChipDesc(chipIds[0]);

    SmallVector<int64_t> workerGridShape = llvm::to_vector(overrideDeviceShape);
    if (workerGridShape.empty()) {
      workerGridShape = llvm::to_vector(device.getWorkerGrid().getShape());
    }

    unsigned dstRegisterSizeTiles = chipDesc.getDstRegisterSizeTiles();
    if (maxDstRegisterSizeTiles.getValue() > 0) {
      dstRegisterSizeTiles =
          std::min(dstRegisterSizeTiles, maxDstRegisterSizeTiles.getValue());
    }

    RewritePatternSet patterns(&getContext());
    patterns.add<TTIRGenericTensorLayoutRewriter>(
        &getContext(), workerGridShape, dstRegisterSizeTiles);
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
      return;
    }
  }

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<mlir::tt::ttir::TTIRDialect>();
    registry.insert<mlir::tt::TTDialect>();
  }
};
} // namespace

} // namespace mlir::tt::ttir
