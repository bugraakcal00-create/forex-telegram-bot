from app.bot import build_application


if __name__ == "__main__":
    app = build_application()
    app.run_polling(drop_pending_updates=True)
