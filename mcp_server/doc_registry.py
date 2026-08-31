"""
doc_registry.py

*** SUPUESTO EXPLÍCITO ***
vectorstore.py guarda en cada chunk metadata = {doc_id, section_title,
char_start, char_end} — no incluye una "categoría" de documento. Pero
guardrails.py necesita categorías para hacer permission scoping ("solo
RRHH puede ver contratos de proveedores", por ejemplo).

En vez de inventar un cambio silencioso al esquema de metadata de Chroma
(que rompería el índice ya construido), este módulo mapea doc_id ->
categoría de forma explícita y auditable, a partir de los nombres de
archivo que ya aparecen en el README del proyecto.

Si tu ingestion/build_index.py ya calcula una categoría en otro lado,
lo correcto es: (a) añadirla a la metadata del chunk en add_chunks(), y
(b) reemplazar esta tabla por una lectura directa de esa metadata. Dejo
ese punto marcado abajo con TODO para que sea una decisión tuya, no mía.

*** IMPORTANTE: formato de doc_id ***
build_index.py define `doc_id = md_file.name`, es decir el doc_id
INCLUYE la extensión ".md" (p. ej. "politica_vacaciones.md", no
"politica_vacaciones"). Las claves de este registro reflejan eso —
si más adelante cambias build_index.py para usar el stem sin
extensión, actualiza también estas claves.
"""

from __future__ import annotations

from dataclasses import dataclass

# doc_id (con extensión ".md", tal como lo genera build_index.py) -> (categoría, título legible)
_REGISTRY: dict[str, tuple[str, str]] = {
    "politica_vacaciones.md": ("politicas", "Política de vacaciones"),
    "politica_home_office.md": ("politicas", "Política de home office"),
    "politica_gastos.md": ("politicas", "Política de gastos"),
    "manual_onboarding.md": ("onboarding", "Manual de onboarding"),
    "contrato_proveedor_cloudtech.md": ("contratos", "Contrato de proveedor — CloudTech"),
    "contrato_proveedor_soporte_ti.md": ("contratos", "Contrato de proveedor — Soporte TI"),
    "faq_soporte_ti.md": ("faq", "FAQ de soporte TI"),
    "codigo_conducta.md": ("conducta", "Código de conducta"),
}

# Categorías válidas, derivadas de la tabla de arriba (para validar inputs).
CATEGORIES = sorted({cat for cat, _ in _REGISTRY.values()})


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    category: str
    title: str


def lookup(doc_id: str) -> DocMeta:
    """
    Devuelve metadata de catálogo para un doc_id.

    TODO(usuario): si build_index.py ya escribe la categoría real como
    metadata del chunk, reemplaza este `_REGISTRY.get(...)` por una
    consulta a esa metadata en vez de a esta tabla estática.
    """
    if doc_id in _REGISTRY:
        category, title = _REGISTRY[doc_id]
        return DocMeta(doc_id=doc_id, category=category, title=title)
    # Documento no catalogado: no rompemos, pero lo marcamos como
    # "sin categoría" para que guardrails.py pueda decidir con qué
    # criterio tratarlo (por defecto, lo más restrictivo).
    return DocMeta(doc_id=doc_id, category="sin_categoria", title=doc_id)


def all_known_doc_ids() -> list[str]:
    return sorted(_REGISTRY.keys())
