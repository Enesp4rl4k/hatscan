# Hatscan

**Elektrik diregi ve iletken hattini fotograftan kontrol eden basit yazilim.**

Drone sart degil — telefon veya drone fotografi yeterli. MVP sentetik gorselle calisir.

Iliskili projeler: [KARGA](https://github.com/Enesp4rl4k/karga) (tam platform), [ADE-R](https://github.com/Enesp4rl4k/ade-r) (karar motoru).

---

## Ne bakar?

| Durum | Uyari kodu |
|--------|------------|
| Direk egilmis | `POLE_TILT` |
| Tel sarkmis | `WIRE_SAG` |
| Tel kopuk gorunuyor | `WIRE_BREAK` |

Cikti: `outputs/report.png` (isaretli foto) + `report.txt` + `report.json`

---

## Kurulum ve calistirma

```bash
pip install -e .
python scripts/run_demo.py
```

Kendi fotografin:

```bash
python scripts/run_demo.py --image yol/foto.jpg
```

Ornek senaryolar:

```bash
python scripts/run_demo.py --tilt 15          # egik direk
python scripts/run_demo.py --broken --tilt 0  # kopuk tel
```

Test:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

---

## Sonraki adimlar

- YOLO ile direk / izolator modeli
- Gercek hat veri seti
- GIS koordinati (EXIF GPS)

---

## Lisans

MIT
