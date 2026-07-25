import ssl

from app.services.tls_service import system_ssl_context


def test_system_ssl_context_keeps_certificate_verification_enabled():
    context = system_ssl_context()

    assert isinstance(context, ssl.SSLContext)
    assert context.verify_mode == ssl.CERT_REQUIRED
    assert context.check_hostname is True
    assert system_ssl_context() is context
