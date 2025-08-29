// app/api/chat/route.ts
import { NextRequest, NextResponse } from 'next/server';

const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE ||
  "http://localhost:7860";

export async function POST(req: NextRequest) {
  // 前端直接传 { question, doc_id, k, min_sim, max_ctx_chars }
  const body = await req.json().catch(() => ({}));
  console.log("proxy ->", API_BASE, "/ask body:", body); // 关键日志
  const r = await fetch(`${API_BASE.replace(/\/$/, "")}/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const text = await r.text();
  return new Response(text, {
    status: r.status,
    headers: { "Content-Type": r.headers.get("Content-Type") || "application/json" },
  });
}

// export async function POST(request: NextRequest) {
//   try {
//     const body = await request.json();

//     const response = await fetch('https://api.openai.com/v1/chat/completions', {
//       method: 'POST',
//       headers: {
//         'Content-Type': 'application/json',
//         'Authorization': `Bearer ${process.env.OPENAI_API_KEY}`,
//       },
//       body: JSON.stringify(body),
//     });

//     if (!response.ok) {
//       throw new Error(`OpenAI API error: ${response.status}`);
//     }

//     const data = await response.json();
//     return NextResponse.json(data);
//   } catch (error) {
//     console.error('OpenAI API Error:', error);
//     return NextResponse.json(
//       { error: 'Failed to process chat request' },
//       { status: 500 }
//     );
//   }
// }
