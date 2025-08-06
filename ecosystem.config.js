module.exports = {
  apps: [
    {
      name: "stock-dashboard",
      script: "app.py",
      interpreter: "python3",
      env: {
        FLASK_ENV: "production",
        FLASK_APP: "app.py"
      }
    }
  ]
}