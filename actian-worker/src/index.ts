import { Container } from "@cloudflare/containers";

export class Actian extends Container {
  defaultPort = 8080;
}

export default {
  async fetch(req: Request, env: any): Promise<Response> {
    const url = new URL(req.url);

    // Route format: /i/<instanceId>/anything
    let instanceId = "global";
    const parts = url.pathname.split("/").filter(Boolean);
    if (parts.length >= 2 && parts[0] === "i") {
      instanceId = parts[1];
      // strip /i/<instanceId> so backend still sees /health, /collections, etc.
      url.pathname = "/" + parts.slice(2).join("/");
      if (url.pathname === "/") url.pathname = "/health";
      req = new Request(url.toString(), req);
    }

    const key = `inst:${instanceId}`;
    const id = env.ACTIAN.idFromName(key);
    const stub = env.ACTIAN.get(id);
    return stub.fetch(req);
  },
};
