#!/bin/bash

# Script per installare le librerie necessarie alla build di un pacchetto Python

set -e

echo "Aggiornamento di pip, setuptools e wheel..."
pip install --upgrade pip setuptools wheel build twine

echo "Installazione completata. Ora puoi procedere con la build del tuo pacchetto Python."