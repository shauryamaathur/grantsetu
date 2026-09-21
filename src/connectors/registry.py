"""Maps a GrantSource.connector_type string to its connector implementation."""
from src.connectors.base import GrantSourceConnector
from src.connectors.grants_gov_connector import GrantsGovAPIConnector
from src.connectors.listing_page_connector import ListingPageDiscoveryConnector
from src.connectors.pdf_listing_connector import PDFListingDiscoveryConnector
from src.connectors.sample_connector import SampleFixtureConnector
from src.connectors.web_discovery_connector import WebDiscoveryConnector

CONNECTOR_REGISTRY: dict[str, type[GrantSourceConnector]] = {
    SampleFixtureConnector.connector_type: SampleFixtureConnector,
    GrantsGovAPIConnector.connector_type: GrantsGovAPIConnector,
    WebDiscoveryConnector.connector_type: WebDiscoveryConnector,
    ListingPageDiscoveryConnector.connector_type: ListingPageDiscoveryConnector,
    PDFListingDiscoveryConnector.connector_type: PDFListingDiscoveryConnector,
}


def get_connector(connector_type: str, config: dict | None = None) -> GrantSourceConnector:
    cls = CONNECTOR_REGISTRY.get(connector_type)
    if cls is None:
        raise ValueError(f"Unknown connector_type '{connector_type}'. Known: {list(CONNECTOR_REGISTRY)}")
    return cls(config)
