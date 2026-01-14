import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'MusaChat - Shared Memory AI',
  description: 'AI assistant with shared memory capabilities',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}

