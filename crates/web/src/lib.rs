//! Langtrain Studio web UI.
//!
//! The UI is plain HTML/CSS/JS with no build step, embedded into the binary
//! at compile time. The daemon serves it on the same port as the API, so the
//! browser, the CLI and the Tauri desktop shell all show the same app.

use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::Router;

struct Asset {
    path: &'static str,
    mime: &'static str,
    body: &'static str,
}

macro_rules! asset {
    ($path:literal, $mime:literal) => {
        Asset { path: $path, mime: $mime, body: include_str!(concat!("../assets/", $path)) }
    };
}

const ASSETS: &[Asset] = &[
    asset!("index.html", "text/html; charset=utf-8"),
    asset!("app.css", "text/css; charset=utf-8"),
    asset!("app.js", "text/javascript; charset=utf-8"),
    asset!("icon.svg", "image/svg+xml"),
];

/// Content hash used as the ETag, so browsers revalidate cheaply after upgrades.
fn etag(body: &str) -> String {
    // FNV-1a: stable across builds and platforms.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in body.bytes() {
        h = (h ^ u64::from(b)).wrapping_mul(0x0100_0000_01b3);
    }
    format!("\"{h:016x}\"")
}

fn serve(path: &str, headers: &HeaderMap) -> Response {
    let Some(asset) = ASSETS.iter().find(|a| a.path == path) else {
        return StatusCode::NOT_FOUND.into_response();
    };
    let tag = etag(asset.body);
    if headers.get(header::IF_NONE_MATCH).and_then(|v| v.to_str().ok()) == Some(tag.as_str()) {
        return StatusCode::NOT_MODIFIED.into_response();
    }
    (
        [
            (header::CONTENT_TYPE, HeaderValue::from_static(asset.mime)),
            (header::CACHE_CONTROL, HeaderValue::from_static("no-cache")),
            (header::ETAG, HeaderValue::from_str(&tag).expect("ascii")),
        ],
        asset.body,
    )
        .into_response()
}

/// Routes for the UI: `/` and `/assets/*`. Merge into the API router.
pub fn router<S: Clone + Send + Sync + 'static>() -> Router<S> {
    Router::new()
        .route("/", get(|h: HeaderMap| async move { serve("index.html", &h) }))
        .route(
            "/assets/{file}",
            get(|axum::extract::Path(file): axum::extract::Path<String>, h: HeaderMap| async move { serve(&file, &h) }),
        )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    #[tokio::test]
    async fn serves_index_and_assets_with_etags() {
        let app: Router = router();
        let res = app.clone().oneshot(Request::get("/").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(res.status(), StatusCode::OK);
        let tag = res.headers()[header::ETAG].clone();

        let res = app
            .clone()
            .oneshot(Request::get("/").header(header::IF_NONE_MATCH, tag).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(res.status(), StatusCode::NOT_MODIFIED);

        for f in ["app.js", "app.css", "icon.svg"] {
            let res = app.clone().oneshot(Request::get(format!("/assets/{f}")).body(Body::empty()).unwrap()).await.unwrap();
            assert_eq!(res.status(), StatusCode::OK, "{f}");
        }
        let res = app.oneshot(Request::get("/assets/nope.js").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(res.status(), StatusCode::NOT_FOUND);
    }
}
