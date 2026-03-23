from app.main import create_app

app = create_app(api_prefix="", include_frontend=False)
