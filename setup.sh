#!/bin/bash
# ─────────────────────────────────────────────────────────────────
#  NutriBot — Script de instalación automática
#  CINF104 Aprendizaje de Máquinas · Grupo 5 · UNAB 2026
#  Uso: bash setup.sh
# ─────────────────────────────────────────────────────────────────

set -e

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   NutriBot — Setup Automático          ${NC}"
echo -e "${GREEN}   CINF104 · Grupo 5 · UNAB 2026        ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""

# ── 1. Detectar SO ──────────────────────────────────────────────
OS="$(uname -s)"
echo -e "${YELLOW}[1/4] Detectando sistema operativo...${NC}"
echo "      Sistema: $OS"

# ── 2. Instalar Ollama ──────────────────────────────────────────
echo ""
echo -e "${YELLOW}[2/4] Verificando Ollama...${NC}"

if command -v ollama &> /dev/null; then
    echo -e "      ${GREEN}✓ Ollama ya está instalado.${NC}"
else
    echo "      Ollama no encontrado. Instalando..."
    if [ "$OS" = "Linux" ] || [ "$OS" = "Darwin" ]; then
        curl -fsSL https://ollama.com/install.sh | sh
        echo -e "      ${GREEN}✓ Ollama instalado.${NC}"
    else
        echo -e "      ${RED}Sistema Windows detectado.${NC}"
        echo "      Descarga Ollama manualmente desde: https://ollama.com/download"
        echo "      Luego vuelve a ejecutar este script."
        exit 1
    fi
fi

# ── 3. Descargar modelo ─────────────────────────────────────────
echo ""
echo -e "${YELLOW}[3/4] Descargando modelo Llama 3.2 (~2GB)...${NC}"
echo "      Esto puede tomar varios minutos según tu internet."
echo ""

if ollama list | grep -q "llama3.2"; then
    echo -e "      ${GREEN}✓ Llama 3.2 ya descargado.${NC}"
else
    ollama pull llama3.2
    echo -e "      ${GREEN}✓ Modelo descargado exitosamente.${NC}"
fi

# ── 4. Levantar servidor Ollama ─────────────────────────────────
echo ""
echo -e "${YELLOW}[4/4] Iniciando servidor Ollama...${NC}"

# Verificar si ya está corriendo
if curl -s http://localhost:11434/api/tags &> /dev/null; then
    echo -e "      ${GREEN}✓ Ollama ya está corriendo en localhost:11434${NC}"
else
    echo "      Iniciando Ollama en segundo plano..."
    ollama serve &
    sleep 3
    if curl -s http://localhost:11434/api/tags &> /dev/null; then
        echo -e "      ${GREEN}✓ Servidor iniciado en localhost:11434${NC}"
    else
        echo -e "      ${RED}No se pudo iniciar Ollama automáticamente.${NC}"
        echo "      Ejecuta manualmente: ollama serve"
    fi
fi

# ── Resultado final ─────────────────────────────────────────────
echo ""
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}   ✅ Instalación completa               ${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "  Para usar el chatbot:"
echo ""
echo "  Opción A — Abrir directamente:"
echo "    Haz doble clic en index.html"
echo ""
echo "  Opción B — Con servidor local (recomendado):"
echo "    python3 -m http.server 8080"
echo "    Luego abre: http://localhost:8080"
echo ""
echo "  Para detener Ollama:"
echo "    pkill ollama"
echo ""
