"""Lisa SDK Extensions — optional plugins for external services.

Each subpackage implements kernel abstractions for a specific external service.
They depend only on kernel/ and can be installed selectively.
"""

from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)
