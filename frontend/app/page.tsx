"use client";

import { useState, useCallback } from "react";
import {
  Shield,
  AlertTriangle,
  Upload,
  Terminal,
  FileWarning,
  CheckCircle,
  Loader2,
  Zap,
} from "lucide-react";
import axios from "axios";

interface ThreatDetection {
  offset: number;
  confidence: number;
  threat_type: string;
  entropy_score: number;
}

interface ScanResponse {
  status: string;
  threats: ThreatDetection[];
  total_chunks: number;
  scan_time: number;
}

export default function Home() {
  const [isDragging, setIsDragging] = useState(false);
  const [isScanning, setIsScanning] = useState(false);
  const [scanResults, setScanResults] = useState<ScanResponse | null>(null);
  const [logs, setLogs] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string | null>(null);

  const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

  const addLog = useCallback((message: string) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev, `[${timestamp}] ${message}`]);
  }, []);

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const scanFile = async (file: File) => {
    setIsScanning(true);
    setSelectedFile(file.name);
    setScanResults(null);
    setLogs([]);

    addLog("> Initializing ViperTrace Scanner...");
    addLog(`> Target: ${file.name} (${(file.size / 1024).toFixed(2)} KB)`);
    addLog("> Establishing secure connection to analysis engine...");

    try {
      const formData = new FormData();
      formData.append("file", file);

      addLog("> Uploading file for deep analysis...");

      const response = await axios.post<ScanResponse>(
        `${API_URL}/scan`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );

      addLog("> File uploaded successfully");
      addLog(`> Processing ${response.data.total_chunks} chunks...`);
      addLog("> Running entropy analysis...");
      addLog("> Executing CNN-based threat detection...");

      setScanResults(response.data);

      if (response.data.status === "clean") {
        addLog("> ✓ SCAN COMPLETE: No threats detected");
        addLog(`> Analysis completed in ${response.data.scan_time}s`);
      } else {
        addLog(
          `> ⚠ WARNING: ${response.data.threats.length} threat(s) detected!`
        );
        response.data.threats.forEach((threat, idx) => {
          addLog(
            `> THREAT ${idx + 1}: ${threat.threat_type} (${
              threat.confidence
            }% confidence)`
          );
          addLog(
            `>   └─ Offset: 0x${threat.offset
              .toString(16)
              .toUpperCase()
              .padStart(4, "0")} | Entropy: ${threat.entropy_score}`
          );
        });
        addLog(`> Total scan time: ${response.data.scan_time}s`);
      }
    } catch (error: any) {
      addLog("> ✗ ERROR: Scan failed");
      if (error.response) {
        addLog(
          `> Server error: ${error.response.data.detail || error.message}`
        );
      } else if (error.request) {
        addLog("> Cannot connect to backend. Is it running on port 8000?");
      } else {
        addLog(`> Error: ${error.message}`);
      }
    } finally {
      setIsScanning(false);
    }
  };

  const handleDrop = async (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);

    const file = e.dataTransfer.files[0];
    if (file) {
      await scanFile(file);
    }
  };

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      await scanFile(file);
    }
  };

  return (
    <div className="min-h-screen bg-black text-green-400 font-mono p-8">
      {/* Header */}
      <div className="max-w-7xl mx-auto">
        <div className="flex items-center justify-center mb-8 space-x-3">
          <Shield className="w-12 h-12 text-green-500 animate-pulse" />
          <h1 className="text-5xl font-bold tracking-wider">
            <span className="text-green-500">VIPER</span>
            <span className="text-green-400">TRACE</span>
          </h1>
        </div>

        <p className="text-center text-green-500 mb-2 text-sm tracking-widest">
          ADVANCED MALWARE DETECTION SYSTEM
        </p>
        <div className="flex items-center justify-center space-x-2 mb-8">
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
          <span className="text-xs text-green-600">SYSTEM ONLINE</span>
          <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
        </div>

        {/* Main Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Upload Zone */}
          <div className="border-2 border-green-500 rounded-lg p-6 bg-black/50">
            <div className="flex items-center space-x-2 mb-4 border-b border-green-800 pb-2">
              <Upload className="w-5 h-5" />
              <h2 className="text-xl font-semibold">FILE UPLOAD</h2>
            </div>

            <div
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
              className={`
                border-2 border-dashed rounded-lg p-12 text-center transition-all cursor-pointer
                ${
                  isDragging
                    ? "border-green-400 bg-green-950/30 scale-105"
                    : "border-green-700 hover:border-green-500 hover:bg-green-950/10"
                }
              `}
            >
              <input
                type="file"
                id="file-upload"
                className="hidden"
                onChange={handleFileSelect}
                disabled={isScanning}
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                {isScanning ? (
                  <div className="flex flex-col items-center">
                    <Loader2 className="w-16 h-16 mb-4 animate-spin text-green-400" />
                    <p className="text-green-500 font-semibold">SCANNING...</p>
                    <p className="text-xs text-green-700 mt-2">
                      {selectedFile}
                    </p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <FileWarning className="w-16 h-16 mb-4 text-green-600" />
                    <p className="text-green-500 font-semibold mb-2">
                      DROP FILE HERE OR CLICK TO SELECT
                    </p>
                    <p className="text-xs text-green-700">
                      Supports all file types • Max size: 50MB
                    </p>
                  </div>
                )}
              </label>
            </div>

            {/* Scan Results Summary */}
            {scanResults && (
              <div className="mt-6 border border-green-800 rounded p-4">
                <div className="flex items-center space-x-2 mb-3">
                  {scanResults.status === "clean" ? (
                    <>
                      <CheckCircle className="w-5 h-5 text-green-500" />
                      <span className="font-semibold text-green-500">
                        FILE CLEAN
                      </span>
                    </>
                  ) : (
                    <>
                      <AlertTriangle className="w-5 h-5 text-red-500" />
                      <span className="font-semibold text-red-500">
                        THREATS DETECTED
                      </span>
                    </>
                  )}
                </div>

                <div className="space-y-2 text-sm">
                  <div className="flex justify-between">
                    <span className="text-green-700">Total Chunks:</span>
                    <span>{scanResults.total_chunks}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-green-700">Threats Found:</span>
                    <span
                      className={
                        scanResults.threats.length > 0 ? "text-red-500" : ""
                      }
                    >
                      {scanResults.threats.length}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-green-700">Scan Time:</span>
                    <span>{scanResults.scan_time}s</span>
                  </div>
                </div>

                {/* Threat Details */}
                {scanResults.threats.length > 0 && (
                  <div className="mt-4 pt-4 border-t border-green-900">
                    <div className="flex items-center space-x-2 mb-2">
                      <Zap className="w-4 h-4 text-red-500" />
                      <span className="text-xs font-semibold text-red-500">
                        THREAT ANALYSIS
                      </span>
                    </div>
                    {scanResults.threats.map((threat, idx) => (
                      <div
                        key={idx}
                        className="mb-3 p-2 bg-red-950/20 border border-red-900 rounded text-xs"
                      >
                        <div className="flex justify-between mb-1">
                          <span className="text-red-400 font-semibold">
                            {threat.threat_type}
                          </span>
                          <span className="text-red-500">
                            {threat.confidence}%
                          </span>
                        </div>
                        <div className="text-green-700">
                          Offset: 0x
                          {threat.offset
                            .toString(16)
                            .toUpperCase()
                            .padStart(4, "0")}{" "}
                          | Entropy: {threat.entropy_score}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Terminal Logs */}
          <div className="border-2 border-green-500 rounded-lg p-6 bg-black/50">
            <div className="flex items-center space-x-2 mb-4 border-b border-green-800 pb-2">
              <Terminal className="w-5 h-5" />
              <h2 className="text-xl font-semibold">ANALYSIS LOG</h2>
            </div>

            <div className="bg-black border border-green-900 rounded p-4 h-96 overflow-y-auto font-mono text-sm">
              {logs.length === 0 ? (
                <div className="text-green-700 text-center mt-20">
                  <Terminal className="w-12 h-12 mx-auto mb-3 opacity-30" />
                  <p>Awaiting file upload...</p>
                  <p className="text-xs mt-2">System ready for analysis</p>
                </div>
              ) : (
                <div className="space-y-1">
                  {logs.map((log, idx) => (
                    <div
                      key={idx}
                      className={`
                        ${
                          log.includes("ERROR") || log.includes("✗")
                            ? "text-red-500"
                            : ""
                        }
                        ${
                          log.includes("WARNING") || log.includes("⚠")
                            ? "text-yellow-500"
                            : ""
                        }
                        ${log.includes("✓") ? "text-green-400" : ""}
                      `}
                    >
                      {log}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="mt-8 text-center text-green-700 text-xs">
          <p>ViperTrace v1.0.0 | Powered by TensorFlow & FastAPI</p>
          <p className="mt-1">Deep Learning Malware Detection System</p>
        </div>
      </div>
    </div>
  );
}
