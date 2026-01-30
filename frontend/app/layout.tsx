import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import ErrorBoundary from '@/components/ErrorBoundary'

const inter = Inter({ subsets: ['latin', 'vietnamese'] })

export const metadata: Metadata = {
    title: 'Recruiter AI Assistant',
    description: 'Intelligent CV Screening & Matching ChatBot',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="vi">
            <body className={inter.className}>
                <ErrorBoundary>
                    {children}
                </ErrorBoundary>
            </body>
        </html>
    )
}
