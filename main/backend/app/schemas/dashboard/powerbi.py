from pydantic import BaseModel


class EmbedTokenResponse(BaseModel):
    embed_token: str
    embed_url: str
    report_id: str
