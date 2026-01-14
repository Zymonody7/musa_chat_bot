/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './app/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        background: '#0f172a',
        surface: '#1e293b',
        primary: '#FF6400',
        primaryHover: '#DB5400',
        secondary: '#64748b',
        textMain: '#f1f5f9',
        textMuted: '#94a3b8',
      },
    },
  },
  plugins: [],
};

