import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        nastaliq: ["'Noto Nastaliq Urdu'", "serif"],
        sans: ["'Inter'", "sans-serif"],
        mono: ["'Fira Code'", "monospace"],
      },
      colors: {
        kashmiri: {
          50: '#ecfeff',
          100: '#cffafe',
          500: '#06b6d4',
          900: '#164e63',
          950: '#083344',
        }
      }
    },
  },
  plugins: [],
};

export default config;
