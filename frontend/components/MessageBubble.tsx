'use client'

import { useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Message } from '@/lib/types'
import CandidateCard from './CandidateCard'

interface MessageBubbleProps {
    message: Message
}

export default function MessageBubble({ message }: MessageBubbleProps) {
    const isUser = message.role === 'user'
    const [showSources, setShowSources] = useState(false)
    const hasChunks = message.retrievedChunks && message.retrievedChunks.length > 0

    return (
        <div
            className={`flex ${isUser ? 'justify-end' : 'justify-start'} chat-bubble-enter`}
        >
            <div
                className={`max-w-[80%] ${isUser
                    ? 'bg-gradient-to-br from-primary-500 to-primary-600 text-white'
                    : 'bg-gray-800/80 text-gray-100 border border-gray-700/50'
                    } rounded-2xl px-4 py-3 shadow-lg`}
            >
                {/* Avatar and Name */}
                {!isUser && (
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
                )}

                {/* Message Content with Markdown */}
                <div className="prose prose-invert prose-sm max-w-none break-words">
                    {isUser ? (
                        <p className="whitespace-pre-wrap">{message.content}</p>
                    ) : (
                        <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                                // Headings
                                h3: ({ children }) => (
                                    <h3 className="text-lg font-bold text-white mt-4 mb-2">{children}</h3>
                                ),
                                h4: ({ children }) => (
                                    <h4 className="text-base font-semibold text-gray-200 mt-3 mb-1">{children}</h4>
                                ),
                                // Tables
                                table: ({ children }) => (
                                    <div className="overflow-x-auto my-3">
                                        <table className="w-full text-sm border-collapse">{children}</table>
                                    </div>
                                ),
                                thead: ({ children }) => (
                                    <thead className="bg-gray-700/50">{children}</thead>
                                ),
                                th: ({ children }) => (
                                    <th className="px-3 py-2 text-left text-xs font-semibold text-gray-300 border-b border-gray-600">{children}</th>
                                ),
                                td: ({ children }) => (
                                    <td className="px-3 py-2 text-gray-200 border-b border-gray-700/50">{children}</td>
                                ),
                                // Lists
                                ul: ({ children }) => (
                                    <ul className="list-disc list-inside space-y-1 my-2">{children}</ul>
                                ),
                                ol: ({ children }) => (
                                    <ol className="list-decimal list-inside space-y-1 my-2">{children}</ol>
                                ),
                                li: ({ children }) => (
                                    <li className="text-gray-200">{children}</li>
                                ),
                                // Horizontal rule
                                hr: () => (
                                    <hr className="border-gray-600 my-4" />
                                ),
                                // Bold and emphasis
                                strong: ({ children }) => (
                                    <strong className="font-bold text-white">{children}</strong>
                                ),
                                em: ({ children }) => (
                                    <em className="italic text-gray-300">{children}</em>
                                ),
                                // Paragraphs
                                p: ({ children }) => (
                                    <p className="text-gray-200 mb-2 leading-relaxed">{children}</p>
                                ),
                                // Code
                                code: ({ children }) => (
                                    <code className="bg-gray-700 px-1 py-0.5 rounded text-xs text-primary-300">{children}</code>
                                ),
                            }}
                        >
                            {message.content}
                        </ReactMarkdown>
                    )}
                </div>

                {/* Candidate Cards */}
                {message.candidates && message.candidates.length > 0 && (
                    <div className="mt-4 pt-3 border-t border-gray-700/50">
                        <p className="text-xs font-medium text-gray-400 mb-3">
                            📋 Ứng viên được đề cập:
                        </p>
                        <div className="space-y-2">
                            {message.candidates.map((candidate) => (
                                <CandidateCard
                                    key={candidate.candidate_id}
                                    candidate={candidate}
                                />
                            ))}
                        </div>
                    </div>
                )}

                {/* View Sources Button (Debug) */}
                {!isUser && hasChunks && (
                    <div className="mt-3 pt-2 border-t border-gray-700/30">
                        <button
                            onClick={() => setShowSources(!showSources)}
                            className="flex items-center gap-2 text-xs text-gray-400 hover:text-primary-400 transition-colors"
                        >
                            <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                            </svg>
                            {showSources ? 'Ẩn nguồn' : `Xem nguồn (${message.retrievedChunks?.length})`}
                        </button>

                        {showSources && (
                            <div className="mt-3 space-y-2 max-h-60 overflow-y-auto">
                                {message.retrievedChunks?.map((chunk, idx) => (
                                    <div
                                        key={chunk.chunk_id || idx}
                                        className="bg-gray-900/50 rounded-lg p-3 border border-gray-700/30"
                                    >
                                        <div className="flex items-center justify-between mb-1">
                                            <span className="text-xs font-medium text-primary-400">
                                                {chunk.candidate_name} - {chunk.section}
                                            </span>
                                            <span className="text-xs text-gray-500">
                                                {(chunk.score * 100).toFixed(0)}% match
                                            </span>
                                        </div>
                                        <p className="text-xs text-gray-300 line-clamp-3">
                                            {chunk.content}
                                        </p>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                )}

                {/* Timestamp */}
                <div
                    className={`text-xs mt-2 ${isUser ? 'text-primary-200' : 'text-gray-500'
                        }`}
                >
                    {new Date(message.timestamp).toLocaleTimeString('vi-VN', {
                        hour: '2-digit',
                        minute: '2-digit',
                    })}
                </div>
            </div>
        </div>
    )
}

