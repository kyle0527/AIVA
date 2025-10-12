from __future__ import annotations

from urllib.parse import urljoin

from bs4 import BeautifulSoup
import httpx

from services.aiva_common.schemas import Asset
from services.aiva_common.utils.ids import new_id


class StaticContentParser:
    """Parse HTML content to extract forms and hyperlinks."""

    def extract(
        self, base_url: str, response: httpx.Response
    ) -> tuple[list[Asset], int]:
        assets: list[Asset] = []
        forms = 0
        if "text/html" in response.headers.get("content-type", ""):
            soup = BeautifulSoup(response.text, "lxml")

            # Process forms - use type checking to ensure we have Tag elements
            forms_list = soup.find_all("form")
            for form in forms_list:
                if not hasattr(form, "get") or not hasattr(form, "find_all"):
                    continue  # Skip if not a proper Tag element

                action = form.get("action")
                if isinstance(action, str | type(None)):
                    action_url = action or base_url
                    full = urljoin(base_url, action_url)
                else:
                    continue  # Skip if action is not a string

                params = []
                input_elements = form.find_all("input")
                for input_elem in input_elements:
                    if hasattr(input_elem, "get"):
                        name = input_elem.get("name")
                        if isinstance(name, str):
                            params.append(name)

                assets.append(
                    Asset(
                        asset_id=new_id("asset"),
                        type="URL",
                        value=full,
                        parameters=params if params else None,
                        has_form=True,
                    )
                )
                forms += 1

            # Process links - use type checking to ensure we have Tag elements
            links_list = soup.find_all("a")
            for a in links_list:
                if not hasattr(a, "get"):
                    continue  # Skip if not a proper Tag element

                href = a.get("href")
                if isinstance(href, str):
                    assets.append(
                        Asset(
                            asset_id=new_id("asset"),
                            type="URL",
                            value=urljoin(base_url, href),
                            has_form=False,
                        )
                    )
        return assets, forms
