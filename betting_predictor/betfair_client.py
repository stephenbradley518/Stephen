"""Thin client for the Betfair Exchange API (API-NG, JSON-RPC).

Supports both interactive login (username/password) and the certificate
login flow Betfair recommends for unattended/automated clients. No
credentials are stored on disk by this module; session tokens live only in
memory for the lifetime of the client instance.
"""

from __future__ import annotations

from typing import Any

import requests

from . import config


class BetfairAPIError(RuntimeError):
    """Raised for login failures or JSON-RPC error responses."""


class BetfairClient:
    def __init__(self, app_key: str, session_token: str):
        if not app_key or not session_token:
            raise ValueError("app_key and session_token are required")
        self.app_key = app_key
        self.session_token = session_token

    # ------------------------------------------------------------------
    # Login
    # ------------------------------------------------------------------
    @classmethod
    def login_interactive(
        cls, username: str, password: str, app_key: str
    ) -> "BetfairClient":
        """Non-certificate login. Simpler, but Betfair may prompt for extra
        verification on some accounts — prefer login_certificate for bots."""
        headers = {
            "Accept": "application/json",
            "X-Application": app_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = requests.post(
            config.BETFAIR_IDENTITY_INTERACTIVE_URL,
            headers=headers,
            data={"username": username, "password": password},
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("status") != "SUCCESS":
            raise BetfairAPIError(f"Betfair login failed: {body}")
        return cls(app_key=app_key, session_token=body["token"])

    @classmethod
    def login_certificate(
        cls,
        username: str,
        password: str,
        app_key: str,
        cert_file: str,
        cert_key: str,
    ) -> "BetfairClient":
        headers = {
            "Accept": "application/json",
            "X-Application": app_key,
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = requests.post(
            config.BETFAIR_IDENTITY_CERT_URL,
            headers=headers,
            data={"username": username, "password": password},
            cert=(cert_file, cert_key),
            timeout=15,
        )
        resp.raise_for_status()
        body = resp.json()
        if body.get("loginStatus") != "SUCCESS":
            raise BetfairAPIError(f"Betfair certificate login failed: {body}")
        return cls(app_key=app_key, session_token=body["sessionToken"])

    @classmethod
    def from_env(cls) -> "BetfairClient":
        if not config.has_betfair_credentials():
            raise ValueError(
                "Missing BETFAIR_APP_KEY / BETFAIR_USERNAME / BETFAIR_PASSWORD "
                "environment variables."
            )
        if config.has_cert_login():
            return cls.login_certificate(
                config.BETFAIR_USERNAME,
                config.BETFAIR_PASSWORD,
                config.BETFAIR_APP_KEY,
                config.BETFAIR_CERT_FILE,
                config.BETFAIR_CERT_KEY,
            )
        return cls.login_interactive(
            config.BETFAIR_USERNAME, config.BETFAIR_PASSWORD, config.BETFAIR_APP_KEY
        )

    # ------------------------------------------------------------------
    # JSON-RPC plumbing
    # ------------------------------------------------------------------
    def _rpc(self, method: str, params: dict[str, Any]) -> Any:
        headers = {
            "X-Application": self.app_key,
            "X-Authentication": self.session_token,
            "Content-Type": "application/json",
        }
        payload = {
            "jsonrpc": "2.0",
            "method": f"SportsAPING/v1.0/{method}",
            "params": params,
            "id": 1,
        }
        resp = requests.post(
            config.BETFAIR_BETTING_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=20,
        )
        resp.raise_for_status()
        body = resp.json()
        if "error" in body:
            raise BetfairAPIError(f"Betfair API error calling {method}: {body['error']}")
        return body["result"]

    # ------------------------------------------------------------------
    # Betting API
    # ------------------------------------------------------------------
    def list_event_types(self, filter_: dict | None = None) -> list[dict]:
        return self._rpc("listEventTypes", {"filter": filter_ or {}})

    def list_market_catalogue(
        self,
        filter_: dict,
        market_projection: list[str] | None = None,
        sort: str = "FIRST_TO_START",
        max_results: int = 50,
    ) -> list[dict]:
        return self._rpc(
            "listMarketCatalogue",
            {
                "filter": filter_,
                "marketProjection": market_projection
                or ["EVENT", "RUNNER_DESCRIPTION", "MARKET_START_TIME"],
                "sort": sort,
                "maxResults": max_results,
            },
        )

    def list_market_book(
        self,
        market_ids: list[str],
        price_projection: dict | None = None,
    ) -> list[dict]:
        return self._rpc(
            "listMarketBook",
            {
                "marketIds": market_ids,
                "priceProjection": price_projection
                or {
                    "priceData": ["EX_BEST_OFFERS", "EX_TRADED"],
                    "virtualise": True,
                },
            },
        )

    def logout(self) -> None:
        requests.post(
            "https://identitysso.betfair.com/api/logout",
            headers={"X-Application": self.app_key, "X-Authentication": self.session_token},
            timeout=10,
        )
