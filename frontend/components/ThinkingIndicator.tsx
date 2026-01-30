'use client'

interface ThinkingIndicatorProps {
    message?: string
}

export default function ThinkingIndicator({ message = 'Đang xử lý...' }: ThinkingIndicatorProps) {
    return (
        <div className="flex justify-start chat-bubble-enter">
            <div className="bg-gray-800/80 border border-gray-700/50 rounded-2xl px-4 py-3 shadow-lg">
                <div className="flex items-center gap-2 mb-2 pb-2 border-b border-gray-700/50">
                    <div className="w-6 h-6 rounded-full bg-gradient-to-br from-primary-400 to-accent-400 flex items-center justify-center">
                        <svg
                            className="w-4 h-4 text-white"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
                            />
                        </svg>
                    </div>
                    <span className="text-xs font-medium text-gray-400">AI Assistant</span>
                </div>

                <div className="flex items-center gap-3">
                    <div className="flex items-center gap-1.5">
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                        <div className="typing-dot"></div>
                    </div>
                    <span className="text-sm text-gray-400">
                        {message}
                    </span>
                </div>
            </div>
        </div>
    )
}
