'use client'

import { useState, useRef, useEffect } from 'react'
import ChatWindow from '@/components/ChatWindow'
import BatchImportModal from '@/components/BatchImportModal'
import SuggestedPrompts from '@/components/SuggestedPrompts'
import { Message, CandidateCard as CandidateCardType, RetrievedChunk } from '@/lib/types'
import { sendChatMessage } from '@/lib/api'
import { v4 as uuidv4 } from 'uuid'

// Generate a unique session ID for this browser session
// Using localStorage instead of sessionStorage to persist across browser restarts
const getSessionId = () => {
    if (typeof window === 'undefined') return ''
    let sessionId = localStorage.getItem('chat_session_id')
    if (!sessionId) {
        sessionId = uuidv4()
        localStorage.setItem('chat_session_id', sessionId)
    }
    return sessionId
}

export default function Home() {
    const [messages, setMessages] = useState<Message[]>([])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [isThinking, setIsThinking] = useState(false)
    const [thinkingStatus, setThinkingStatus] = useState('Đang xử lý...')
    const [isImportModalOpen, setIsImportModalOpen] = useState(false)
    const [sessionId, setSessionId] = useState('')
    const abortControllerRef = useRef<AbortController | null>(null)

    useEffect(() => {
        setSessionId(getSessionId())
    }, [])

    const handleSendMessage = async (messageText: string) => {
        if (!messageText.trim() || isLoading || !sessionId) return

        const userMessage: Message = {
            id: uuidv4(),
            role: 'user',
            content: messageText,
            timestamp: new Date(),
        }

        setMessages((prev) => [...prev, userMessage])
        setInput('')
        setIsLoading(true)
        setIsThinking(true)

        // Create assistant message placeholder
        const assistantMessageId = uuidv4()
        const assistantMessage: Message = {
            id: assistantMessageId,
            role: 'assistant',
            content: '',
            timestamp: new Date(),
            candidates: [],
            retrievedChunks: [],
        }
        setMessages((prev) => [...prev, assistantMessage])

        try {
            abortControllerRef.current = new AbortController()

            await sendChatMessage(
                sessionId,
                messageText,
                // On token received
                (token: string) => {
                    setIsThinking(false)
                    setMessages((prev) =>
                        prev.map((msg) =>
                            msg.id === assistantMessageId
                                ? { ...msg, content: msg.content + token }
                                : msg
                        )
                    )
                },
                // On candidates received
                (candidates: CandidateCardType[]) => {
                    setMessages((prev) =>
                        prev.map((msg) =>
                            msg.id === assistantMessageId
                                ? { ...msg, candidates }
                                : msg
                        )
                    )
                },
                // On error
                (error: string) => {
                    setMessages((prev) =>
                        prev.map((msg) =>
                            msg.id === assistantMessageId
                                ? { ...msg, content: `Lỗi: ${error}` }
                                : msg
                        )
                    )
                },
                // On status received
                (status: string) => {
                    setThinkingStatus(status)
                },
                abortControllerRef.current.signal,
                // On chunks received (debug info)
                (chunks: RetrievedChunk[]) => {
                    setMessages((prev) =>
                        prev.map((msg) =>
                            msg.id === assistantMessageId
                                ? { ...msg, retrievedChunks: chunks }
                                : msg
                        )
                    )
                }
            )

        } catch (error) {
            console.error('Chat error:', error)
        } finally {
            setIsLoading(false)
            setIsThinking(false)
            setThinkingStatus('Đang xử lý...')
            abortControllerRef.current = null
        }
    }

    const handleSuggestedPrompt = (prompt: string) => {
        handleSendMessage(prompt)
    }

    const handleNewChat = () => {
        // Clear messages and create new session
        setMessages([])
        const newSessionId = uuidv4()
        localStorage.setItem('chat_session_id', newSessionId)
        setSessionId(newSessionId)
    }

    return (
        <main className="flex flex-col h-screen bg-gradient-to-br from-gray-900 via-gray-800 to-gray-900">
            {/* Batch Import Modal */}
            <BatchImportModal
                isOpen={isImportModalOpen}
                onClose={() => setIsImportModalOpen(false)}
            />

            {/* Header */}
            <header className="flex items-center justify-between px-6 py-4 border-b border-gray-700/50 glass-dark">
                <div className="flex items-center gap-3">
                    <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-primary-500 to-accent-500 flex items-center justify-center">
                        <svg
                            className="w-6 h-6 text-white"
                            fill="none"
                            viewBox="0 0 24 24"
                            stroke="currentColor"
                        >
                            <path
                                strokeLinecap="round"
                                strokeLinejoin="round"
                                strokeWidth={2}
                                d="M8 12h.01M12 12h.01M16 12h.01M21 12c0 4.418-4.03 8-9 8a9.863 9.863 0 01-4.255-.949L3 20l1.395-3.72C3.512 15.042 3 13.574 3 12c0-4.418 4.03-8 9-8s9 3.582 9 8z"
                            />
                        </svg>
                    </div>
                    <div>
                        <h1 className="text-xl font-bold gradient-text">
                            Recruiter AI Assistant
                        </h1>
                        <p className="text-xs text-gray-400">
                            Tìm kiếm & Đánh giá ứng viên thông minh
                        </p>
                    </div>
                </div>

                <div className="flex items-center gap-3">
                    <button
                        onClick={() => setIsImportModalOpen(true)}
                        className="px-4 py-2 text-sm font-medium text-gray-300 hover:text-white bg-gray-700/50 hover:bg-gray-600/50 rounded-lg transition-all duration-200 flex items-center gap-2"
                    >
                        <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                        </svg>
                        Batch Import
                    </button>

                    <button
                        onClick={handleNewChat}
                        className="px-4 py-2 text-sm font-medium text-gray-300 hover:text-white bg-gray-700/50 hover:bg-gray-600/50 rounded-lg transition-all duration-200"
                    >
                        + Cuộc trò chuyện mới
                    </button>
                </div>
            </header>

            {/* Chat Area */}
            <div className="flex-1 overflow-hidden">
                {messages.length === 0 ? (
                    <div className="h-full flex flex-col items-center justify-center p-6">
                        <div className="text-center mb-8">
                            <h2 className="text-3xl font-bold text-white mb-2">
                                Xin chào! 👋
                            </h2>
                            <p className="text-gray-400 max-w-md">
                                Tôi là AI Assistant giúp bạn tìm kiếm và đánh giá ứng viên.
                                Hãy bắt đầu bằng một câu hỏi!
                            </p>
                        </div>
                        <SuggestedPrompts onSelectPrompt={handleSuggestedPrompt} />
                    </div>
                ) : (
                    <ChatWindow
                        messages={messages}
                        isThinking={isThinking}
                        thinkingStatus={thinkingStatus}
                    />
                )}
            </div>

            {/* Input Area */}
            <div className="p-4 border-t border-gray-700/50 glass-dark">
                <div className="max-w-4xl mx-auto">
                    <form
                        onSubmit={(e) => {
                            e.preventDefault()
                            handleSendMessage(input)
                        }}
                        className="flex gap-3"
                    >
                        <input
                            type="text"
                            value={input}
                            onChange={(e) => setInput(e.target.value)}
                            placeholder="Hỏi về ứng viên, tìm kiếm theo kỹ năng, kinh nghiệm..."
                            className="flex-1 px-4 py-3 bg-gray-800/80 border border-gray-600/50 rounded-xl text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500/50 focus:border-primary-500/50 transition-all duration-200"
                            disabled={isLoading}
                        />
                        <button
                            type="submit"
                            disabled={isLoading || !input.trim()}
                            className="px-6 py-3 bg-gradient-to-r from-primary-500 to-accent-500 text-white font-medium rounded-xl hover:from-primary-600 hover:to-accent-600 disabled:opacity-50 disabled:cursor-not-allowed transition-all duration-200 flex items-center gap-2"
                        >
                            {isLoading ? (
                                <svg
                                    className="w-5 h-5 animate-spin"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                >
                                    <circle
                                        className="opacity-25"
                                        cx="12"
                                        cy="12"
                                        r="10"
                                        stroke="currentColor"
                                        strokeWidth="4"
                                    />
                                    <path
                                        className="opacity-75"
                                        fill="currentColor"
                                        d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                                    />
                                </svg>
                            ) : (
                                <svg
                                    className="w-5 h-5"
                                    fill="none"
                                    viewBox="0 0 24 24"
                                    stroke="currentColor"
                                >
                                    <path
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                        strokeWidth={2}
                                        d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8"
                                    />
                                </svg>
                            )}
                            Gửi
                        </button>
                    </form>
                </div>
            </div>
        </main>
    )
}
