import json
import requests


def create(args):

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer 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",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = json.dumps({"title": args.issue, "body": args.binary})

    response = requests.post(
        f"https://api.github.com/repos/ttdloke/mlir_workflows/issues",
        headers=headers,
        data=data,
    )

    res = response.json()
    print("Here is a link to the issue just created: " + res["html_url"])


def upload(args):

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer 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",
        "X-GitHub-Api-Version": "2022-11-28",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    data = json.dumps({"body": args.binary})

    response = requests.patch(
        f"https://api.github.com/repos/ttdloke/mlir_workflows/issues/{args.issue}",
        headers=headers,
        data=data,
    )

    res = response.json()
    print("Here is a link to the issue just updated: " + res["html_url"])


def download(args):

    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": "Bearer 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",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    response = requests.get(
        f"https://api.github.com/repos/ttdloke/mlir_workflows/issues/{args.issue}",
        headers=headers,
    )

    res = response.json()
    f = open(args.binary, "w")
    f.write(str(res))
    f.close()
    print("The file was just downloaded to: " + args.binary)
