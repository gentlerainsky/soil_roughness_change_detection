import ee
from config import (
    SERVICE_ACCOUNT,
    CREDENTIAL_PATH
)

def authenticate(
        service_account: str = SERVICE_ACCOUNT,
        credential_path: str = CREDENTIAL_PATH
    ):
    credentials = ee.ServiceAccountCredentials(service_account, credential_path)
    ee.Initialize(credentials)
