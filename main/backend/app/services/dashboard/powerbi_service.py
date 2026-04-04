import logging

import httpx
from msal import ConfidentialClientApplication
from pydantic import BaseModel

from app.core.settings import get_powerbi_settings

logger = logging.getLogger(__name__)

MOCK_ACCESS_TOKEN = "mock_access_token"
MOCK_EMBED_TOKEN = "mock_embed_token_for_local_testing"


class EmbedTokenResponse(BaseModel):
    embed_token: str
    embed_url: str
    report_id: str


class PowerBIService:
    def __init__(self) -> None:
        self.settings = get_powerbi_settings()
        self.authority_url = f"https://login.microsoftonline.com/{self.settings.tenant_id}"
        self.scope = ["https://analysis.windows.net/powerbi/api/.default"]

    def _get_access_token(self) -> str | None:
        if not self.settings.client_id or not self.settings.client_secret or not self.settings.tenant_id:
            logger.warning("Power BI settings are missing. Returning mock access token.")
            return MOCK_ACCESS_TOKEN

        client_app = ConfidentialClientApplication(
            self.settings.client_id,
            client_credential=self.settings.client_secret,
            authority=self.authority_url,
        )

        result = client_app.acquire_token_silent(self.scope, account=None)
        if not result:
            logger.info("No cached token found. Fetching token from Azure AD.")
            result = client_app.acquire_token_for_client(scopes=self.scope)

        if "access_token" in result:
            return result["access_token"]

        logger.error("Failed to acquire token: %s", result.get("error_description", "Unknown error"))
        return None

    def get_embed_info(self) -> EmbedTokenResponse:
        """
        Fetch embed token and embed URL from the Power BI REST API.
        Returns mock values when credentials are not configured.
        """
        if not self.settings.workspace_id or not self.settings.report_id:
            return EmbedTokenResponse(
                embed_token=MOCK_EMBED_TOKEN,
                embed_url="https://app.powerbi.com/reportEmbed?reportId=mock_report_id",
                report_id="mock_report_id",
            )

        access_token = self._get_access_token()
        if not access_token or access_token == MOCK_ACCESS_TOKEN:
            return EmbedTokenResponse(
                embed_token=MOCK_EMBED_TOKEN,
                embed_url=f"https://app.powerbi.com/reportEmbed?reportId={self.settings.report_id}",
                report_id=self.settings.report_id,
            )

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}",
        }
        report_url = (
            "https://api.powerbi.com/v1.0/myorg/groups/"
            f"{self.settings.workspace_id}/reports/{self.settings.report_id}"
        )

        try:
            with httpx.Client(timeout=30.0) as client:
                report_response = client.get(report_url, headers=headers)
                report_response.raise_for_status()
                report_data = report_response.json()
                embed_url = report_data.get("embedUrl", "")

                token_url = (
                    "https://api.powerbi.com/v1.0/myorg/groups/"
                    f"{self.settings.workspace_id}/reports/{self.settings.report_id}/GenerateToken"
                )
                body = {"accessLevel": "View"}
                token_response = client.post(token_url, headers=headers, json=body)
                token_response.raise_for_status()
                token_data = token_response.json()
                embed_token = token_data.get("token", "")

                return EmbedTokenResponse(
                    embed_token=embed_token,
                    embed_url=embed_url,
                    report_id=self.settings.report_id,
                )
        except httpx.HTTPStatusError as exc:
            logger.error("HTTP error while fetching Power BI embed info: %s", exc)
            logger.error("Power BI response body: %s", exc.response.text)
        except Exception as exc:  # noqa: BLE001
            logger.error("Unexpected error while fetching Power BI embed info: %s", exc)

        return EmbedTokenResponse(
            embed_token="error_fetching_token",
            embed_url=f"https://app.powerbi.com/reportEmbed?reportId={self.settings.report_id}",
            report_id=self.settings.report_id,
        )


def get_powerbi_service() -> PowerBIService:
    return PowerBIService()
