const { createProxyMiddleware } = require("http-proxy-middleware");

/** CRA dev server forwards /api/* to the franchise backend (avoids browser CORS). */
module.exports = function setupProxy(app) {
  const target = process.env.REACT_APP_API_PROXY || "http://127.0.0.1:8000";

  app.use(
    "/api",
    createProxyMiddleware({
      target,
      changeOrigin: true,
      logLevel: "warn",
    })
  );
};
