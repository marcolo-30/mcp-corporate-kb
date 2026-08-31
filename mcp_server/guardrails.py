"""
guardrails.py

Dos responsabilidades separadas a propósito (no mezclar autorización
con logging en el mismo call site, para que cada una se pueda testear
y auditar de forma independiente):

1. Permission scoping por categoría de documento — qué "rol" de MCP
   client puede tocar qué categoría.
2. Structured audit logging — cada llamada a un tool queda registrada
   en JSON lines, con qué se pidió, qué se devolvió (resumen, no el
   contenido completo) y si fue autorizado o denegado.

*** SUPUESTO EXPLÍCITO ***
El protocolo MCP no tiene un campo estándar de "usuario/rol" en la
llamada a un tool. Aquí se modela como un parámetro opcional `role`
que el cliente MCP (agent_client/agent.py) puede pasar en cada tool
call; si no lo pasa, se usa config.DEFAULT_ROLE. En un despliegue real
esto normalmente vendría de la capa de auth del transporte (p. ej.
un token verificado en el handshake), no de un argumento de la tool —
pero como el server corre por stdio en este proyecto, no hay handshake
de auth que lo provea gratis. Documentado en vez de resuelto en silencio.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from mcp_server import config
from mcp_server.doc_registry import CATEGORIES

# Matriz de permisos: rol -> categorías visibles.
# "contratos" queda restringido por defecto porque son documentos legales
# con terceros; el resto es visible para cualquier empleado.
_ROLE_PERMISSIONS: dict[str, set[str]] = {
    "employee": {"politicas", "onboarding", "faq", "conducta"},
    "hr": set(CATEGORIES) | {"sin_categoria"},
    "legal": set(CATEGORIES) | {"sin_categoria"},
    "admin": set(CATEGORIES) | {"sin_categoria"},
}


class PermissionDenied(Exception):
    def __init__(self, role: str, category: str):
        self.role = role
        self.category = category
        super().__init__(
            f"El rol '{role}' no tiene acceso a documentos de categoría '{category}'."
        )


def check_category_access(category: str, role: str | None = None) -> None:
    """Lanza PermissionDenied si el rol no puede ver esa categoría."""
    role = role or config.DEFAULT_ROLE
    allowed = _ROLE_PERMISSIONS.get(role, _ROLE_PERMISSIONS["employee"])
    if category not in allowed:
        raise PermissionDenied(role=role, category=category)


def filter_allowed_categories(role: str | None = None) -> set[str]:
    role = role or config.DEFAULT_ROLE
    return _ROLE_PERMISSIONS.get(role, _ROLE_PERMISSIONS["employee"])


def log_tool_call(
    tool_name: str,
    args: dict,
    role: str | None,
    result_summary: str,
    authorized: bool = True,
) -> None:
    """
    Escribe una línea JSON por cada llamada a un tool. Se registra un
    resumen del resultado (no el contenido completo del documento) para
    que el log de auditoría sea útil sin duplicar todo el corpus en disco.
    """
    config.AUDIT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": time.time(),
        "tool": tool_name,
        "role": role or config.DEFAULT_ROLE,
        "args": args,
        "authorized": authorized,
        "result_summary": result_summary,
    }
    with Path(config.AUDIT_LOG_PATH).open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
