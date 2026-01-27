"""
Tippmix AI Segéd - Application Entry Point
===========================================
Futtatás: python run.py
"""

from app import create_app

app = create_app()

if __name__ == "__main__":
    print("[*] Tippmix AI Seged inditas...")
    print("[*] Nyisd meg a bongeszodben: http://127.0.0.1:5000")
    print("=" * 50)
    
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
