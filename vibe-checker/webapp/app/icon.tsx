export const size = {
  width: 32,
  height: 32,
};
export const contentType = "image/svg+xml";

export default function Icon() {
  const svg = `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 32 32">
    <defs>
      <style>
        @import url('https://fonts.cdnfonts.com/css/joypixels');
        .emoji {
          font-family: 'JoyPixels', 'Apple Color Emoji', 'Segoe UI Emoji', sans-serif;
          font-size: 24px;
          fill: currentColor;
        }
      </style>
    </defs>
    <text y="24" x="16" text-anchor="middle" class="emoji">🤙</text>
  </svg>`;

  return new Response(svg, {
    headers: {
      "Content-Type": "image/svg+xml",
    },
  });
}
