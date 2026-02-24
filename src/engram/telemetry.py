"""OpenTelemetry integration for engram. Disabled by default; activated via config.

OTel packages are optional extras — import errors are silently swallowed when
the `telemetry` extra is not installed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from engram.config import Config

_tracer = None
_meter = None


def setup_telemetry(config: "Config") -> None:
    """Initialize OTel tracer and meter providers from config.

    No-op when telemetry.enabled is False or OTel packages are not installed.
    """
    global _tracer, _meter

    if not config.telemetry.enabled:
        return

    try:
        from opentelemetry import metrics, trace
        from opentelemetry.sdk.metrics import MeterProvider
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError:
        import logging
        logging.getLogger("engram").warning(
            "opentelemetry packages not installed; telemetry disabled. "
            "Install with: pip install 'engram[telemetry]'"
        )
        return

    provider = TracerProvider()

    if config.telemetry.otlp_endpoint:
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
            provider.add_span_processor(
                BatchSpanProcessor(
                    OTLPSpanExporter(endpoint=config.telemetry.otlp_endpoint)
                )
            )
        except ImportError:
            import logging
            logging.getLogger("engram").warning(
                "OTLP exporter not installed; traces will not be exported. "
                "Install with: pip install 'engram[telemetry]'"
            )

    trace.set_tracer_provider(provider)
    _tracer = trace.get_tracer(config.telemetry.service_name)

    meter_provider = MeterProvider()
    metrics.set_meter_provider(meter_provider)
    _meter = metrics.get_meter(config.telemetry.service_name)


def get_tracer():
    """Return the configured tracer, or None if telemetry is disabled."""
    return _tracer


def get_meter():
    """Return the configured meter, or None if telemetry is disabled."""
    return _meter
