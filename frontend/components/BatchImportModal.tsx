'use client'

import { useState } from 'react'
import { scanDirectory } from '@/lib/api'

interface BatchImportModalProps {
    isOpen: boolean
    onClose: () => void
}

export default function BatchImportModal({ isOpen, onClose }: BatchImportModalProps) {
    const [directory, setDirectory] = useState('./public_cvs')
    const [driveUrl, setDriveUrl] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [result, setResult] = useState<{
        message: string
        found_files: number
        triggered_tasks: number
        errors: string[]
    } | null>(null)
    const [error, setError] = useState('')

    if (!isOpen) return null

    const handleScan = async () => {
        setIsLoading(true)
        setError('')
        setResult(null)

        try {
            const data = await scanDirectory(directory, driveUrl)
            setResult(data)
        } catch (err: any) {
            setError(err.message || 'Có lỗi xảy ra khi quét thư mục')
        } finally {
            setIsLoading(false)
        }
    }

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50 backdrop-blur-sm">
            <div className="w-full max-w-md bg-gray-800 border border-gray-700 rounded-xl shadow-2xl p-6">
                <div className="flex justify-between items-center mb-6">
                    <h2 className="text-xl font-bold text-white">Batch Import CVs</h2>
                    <button
                        onClick={onClose}
                        className="text-gray-400 hover:text-white transition-colors"
                    >
                        <svg className="w-6 h-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                        </svg>
                    </button>
                </div>

                {!result ? (
                    <div className="space-y-4">
                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Google Drive Folder Link (Optional)
                            </label>
                            <input
                                type="text"
                                value={driveUrl}
                                onChange={(e) => setDriveUrl(e.target.value)}
                                placeholder="https://drive.google.com/..."
                                className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
                            />
                            <p className="mt-1 text-xs text-gray-500">
                                Nếu nhập link này, hệ thống sẽ tải file từ Drive về trước khi xử lý.
                            </p>
                        </div>

                        <div className="relative">
                            <div className="absolute inset-0 flex items-center">
                                <div className="w-full border-t border-gray-700"></div>
                            </div>
                            <div className="relative flex justify-center text-sm">
                                <span className="px-2 bg-gray-800 text-gray-500">Hoặc quét thư mục local</span>
                            </div>
                        </div>

                        <div>
                            <label className="block text-sm font-medium text-gray-300 mb-2">
                                Thư mục trên Server
                            </label>
                            <input
                                type="text"
                                value={directory}
                                onChange={(e) => setDirectory(e.target.value)}
                                placeholder="./public_cvs"
                                className="w-full px-4 py-2 bg-gray-900 border border-gray-600 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500/50"
                            />
                            <p className="mt-1 text-xs text-gray-500">
                                Nhập đường dẫn tương đối (từ thư mục project) hoặc tuyệt đối.
                            </p>
                        </div>

                        {error && (
                            <div className="p-3 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
                                {error}
                            </div>
                        )}

                        <button
                            onClick={handleScan}
                            disabled={isLoading}
                            className="w-full py-2 bg-primary-600 hover:bg-primary-700 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex justify-center items-center gap-2"
                        >
                            {isLoading ? (
                                <>
                                    <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                                    </svg>
                                    Beginning Scan...
                                </>
                            ) : (
                                'Scan & Import'
                            )}
                        </button>
                    </div>
                ) : (
                    <div className="space-y-4">
                        <div className="p-4 bg-green-500/10 border border-green-500/20 rounded-lg">
                            <h3 className="text-green-400 font-medium mb-1">✅ Đã bắt đầu xử lý!</h3>
                            <p className="text-gray-300 text-sm">
                                Tìm thấy: <span className="font-bold">{result.found_files}</span> files PDF
                            </p>
                            <p className="text-gray-300 text-sm">
                                Đã trigger: <span className="font-bold">{result.triggered_tasks}</span> tasks
                            </p>
                        </div>

                        {result.errors && result.errors.length > 0 && (
                            <div className="p-3 bg-yellow-500/10 border border-yellow-500/20 rounded-lg max-h-40 overflow-y-auto">
                                <p className="text-yellow-400 text-sm font-medium mb-1">Cảnh báo:</p>
                                <ul className="list-disc list-inside text-xs text-yellow-300/80">
                                    {result.errors.map((err, i) => (
                                        <li key={i}>{err}</li>
                                    ))}
                                </ul>
                            </div>
                        )}

                        <button
                            onClick={() => {
                                setResult(null)
                                onClose()
                            }}
                            className="w-full py-2 bg-gray-700 hover:bg-gray-600 text-white font-medium rounded-lg transition-colors"
                        >
                            Đóng
                        </button>
                    </div>
                )}
            </div>
        </div>
    )
}
