from __future__ import annotations


class ProviderUnavailable(RuntimeError):
    pass


class Provider:
    name = "base"
    capabilities: set[str] = set()
    cost_note = ""

    def available(self) -> tuple[bool, str]:
        return False, "not implemented"

    def require(self) -> None:
        ok, reason = self.available()
        if not ok:
            raise ProviderUnavailable(f"{self.name}: {reason}")
