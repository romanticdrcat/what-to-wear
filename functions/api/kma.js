export async function onRequestGet(context) {
  const { request, env } = context;

  // ✅ Cloudflare Pages 환경변수로 저장해둔 KMA 키
  const serviceKey = env.KMA_SERVICE_KEY;
  if (!serviceKey) {
    return new Response("Missing KMA_SERVICE_KEY in Cloudflare Pages env", { status: 500 });
  }

  const url = new URL(request.url);

  // ✅ 클라이언트가 넘긴 쿼리
  // 예: base_date, base_time, nx, ny, numOfRows, pageNo 등
  const qs = new URLSearchParams(url.searchParams);

  // ✅ key는 프록시에서 강제로 주입 (클라에서 안 받음)
  qs.set("serviceKey", serviceKey);
  qs.set("dataType", qs.get("dataType") || "JSON");
  qs.set("numOfRows", qs.get("numOfRows") || "1000");
  qs.set("pageNo", qs.get("pageNo") || "1");

  const upstream = `https://apis.data.go.kr/1360000/VilageFcstInfoService_2.0/getVilageFcst?${qs.toString()}`;

  // ✅ 타임아웃(Cloudflare는 AbortController 가능)
  const controller = new AbortController();
  const t = setTimeout(() => controller.abort(), 8000);

  try {
    const res = await fetch(upstream, {
      method: "GET",
      signal: controller.signal,
      headers: {
        "Accept": "application/json",
      },
    });

    // 그대로 반환(상태코드 포함)
    const body = await res.text();
    return new Response(body, {
      status: res.status,
      headers: {
        "Content-Type": res.headers.get("content-type") || "application/json; charset=utf-8",
        "Access-Control-Allow-Origin": "*",
        "Cache-Control": "no-store",
      },
    });
  } catch (e) {
    return new Response(`Upstream fetch failed: ${String(e)}`, { status: 502 });
  } finally {
    clearTimeout(t);
  }
}
