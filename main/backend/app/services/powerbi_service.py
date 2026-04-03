import logging
from msal import ConfidentialClientApplication
import httpx
from pydantic import BaseModel

from app.core.settings import get_powerbi_settings

logger = logging.getLogger(__name__)

class EmbedTokenResponse(BaseModel):
    embed_token: str
    embed_url: str
    report_id: str

class PowerBIService:
    def __init__(self):
        self.settings = get_powerbi_settings()
        self.authority_url = f"https://login.microsoftonline.com/{self.settings.tenant_id}"
        self.powerbi_api_url = "https://api.powerbi.com/"
        self.scope = ["https://analysis.windows.net/powerbi/api/.default"]

    def _get_access_token(self) -> str | None:
        if not self.settings.client_id or not self.settings.client_secret or not self.settings.tenant_id:
            logger.warning("PowerBI settings missing. Returning mock access token.")
            return "mock_access_token"

        client_app = ConfidentialClientApplication(
            self.settings.client_id,
            client_credential=self.settings.client_secret,
            authority=self.authority_url,
        )

        result = client_app.acquire_token_silent(self.scope, account=None)
        if not result:
            logger.info("No cached token found. Fetching from authority.")
            result = client_app.acquire_token_for_client(scopes=self.scope)

        if "access_token" in result:
            return result["access_token"]
        else:
            logger.error(f"Failed to acquire token: {result.get('error_description', 'Unknown error')}")
            return None

    def get_embed_info(self) -> EmbedTokenResponse:
        """
        Fetches Embed Token and Embed URL from Power BI REST API.
        If credentials are not set in .env, returns mock data for UI testing.
        """
        # If no config is provided, return placeholder for local testing
        if not self.settings.workspace_id or not self.settings.report_id:
            return EmbedTokenResponse(
                embed_token="mock_embed_token_for_local_testing",
                embed_url="https://app.powerbi.com/reportEmbed?reportId=mock_report_id",
                report_id="mock_report_id"
            )

        access_token = self._get_access_token()
        if not access_token or access_token == "mock_access_token":
            return EmbedTokenResponse(
                embed_token="mock_embed_token_for_local_testing",
                embed_url=f"https://app.powerbi.com/reportEmbed?reportId={self.settings.report_id}",
                report_id=self.settings.report_id
            )

        # Actual Power BI REST API Call
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {access_token}"
        }
        
        # 1. Get Report details (includes Embed URL)
        report_url = f"https://api.powerbi.com/v1.0/myorg/groups/{self.settings.workspace_id}/reports/{self.settings.report_id}"
        
        try:
            # We use httpx.Client here to be synchronous, matching the FastAPI standard dependencies
            with httpx.Client() as client:
                report_response = client.get(report_url, headers=headers)
                report_response.raise_for_status()
                report_data = report_response.json()
                embed_url = report_data.get("embedUrl", "")

                # 2. Get Embed Token
                token_url = f"https://api.powerbi.com/v1.0/myorg/groups/{self.settings.workspace_id}/reports/{self.settings.report_id}/GenerateToken"
                body = {
                    "accessLevel": "View"
                }
                token_response = client.post(token_url, headers=headers, json=body)
                token_response.raise_for_status()
                token_data = token_response.json()
                embed_token = token_data.get("token", "")

                return EmbedTokenResponse(
                    embed_token=embed_token,
                    embed_url=embed_url,
                    report_id=self.settings.report_id
                )
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error fetching Power BI embed info: {e}")
            logger.error(f"Response body: {e.response.text}")
            # Fallback to mock on error so application doesn't crash during local demo
            return EmbedTokenResponse(
                embed_token="error_fetching_token",
                embed_url=f"https://app.powerbi.com/reportEmbed?reportId={self.settings.report_id}",
                report_id=self.settings.report_id
            )
        except Exception as e:
            logger.error(f"Unexpected error fetching Power BI embed info: {e}")
            return EmbedTokenResponse(
                embed_token="error_fetching_token",
                embed_url=f"https://app.powerbi.com/reportEmbed?reportId={self.settings.report_id}",
                report_id=self.settings.report_id
            )

def get_powerbi_service() -> PowerBIService:
    return PowerBIService()
