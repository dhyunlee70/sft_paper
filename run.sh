#!/bin/bash
# SFT 논문 결과 테이블 생성
# 실행: bash run.sh

cd "$(dirname "$0")"
echo "=================================================="
echo " SFT Paper Tables 생성 시작"
echo "=================================================="
python3 make_tables.py
echo ""
echo "결과 파일:"
ls output/
