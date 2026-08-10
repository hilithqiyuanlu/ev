import Foundation

enum JSONValue: Codable, Hashable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() { self = .null }
        else if let value = try? container.decode(Bool.self) { self = .bool(value) }
        else if let value = try? container.decode(Double.self) { self = .number(value) }
        else if let value = try? container.decode(String.self) { self = .string(value) }
        else if let value = try? container.decode([String: JSONValue].self) { self = .object(value) }
        else { self = .array(try container.decode([JSONValue].self)) }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case .string(let value): try container.encode(value)
        case .number(let value): try container.encode(value)
        case .bool(let value): try container.encode(value)
        case .object(let value): try container.encode(value)
        case .array(let value): try container.encode(value)
        case .null: try container.encodeNil()
        }
    }

    var string: String? { if case .string(let value) = self { value } else { nil } }
    var double: Double? { if case .number(let value) = self { value } else { nil } }
    var bool: Bool? { if case .bool(let value) = self { value } else { nil } }
    var object: [String: JSONValue]? { if case .object(let value) = self { value } else { nil } }
    var array: [JSONValue]? { if case .array(let value) = self { value } else { nil } }
}

struct EngineEnvelope: Decodable {
    let version: Int
    let requestID: String?
    let type: String
    let timestamp: String
    let payload: [String: JSONValue]

    enum CodingKeys: String, CodingKey {
        case version, type, timestamp, payload
        case requestID = "request_id"
    }
}

struct AudioDevice: Identifiable, Hashable {
    let index: Int
    let name: String
    let isDefault: Bool
    var id: Int { index }

    init?(_ value: JSONValue) {
        guard let object = value.object,
              let index = object["index"]?.double,
              let name = object["name"]?.string else { return nil }
        self.index = Int(index)
        self.name = name
        self.isDefault = object["is_default"]?.bool ?? false
    }
}

struct Segment: Identifiable, Hashable {
    let id: String
    let startedAt: String
    let durationMS: Int
    let audioPath: String
    let transcript: String
    let speakerLabel: String
    let speakerScore: Double?
    let wakeDetected: Bool
    let queryCandidate: Bool
    let queryText: String
    let wasCorrected: Bool

    var speakerDisplayLabel: String {
        switch speakerLabel {
        case "user": return "我"
        case "non-user": return "他人"
        default: return speakerLabel
        }
    }

    var sortDate: String { startedAt }

    init?(_ object: [String: JSONValue]) {
        guard let id = object["id"]?.string,
              let startedAt = object["started_at"]?.string,
              let audioPath = object["audio_path"]?.string else { return nil }
        self.id = id
        self.startedAt = startedAt
        self.durationMS = Int(object["duration_ms"]?.double ?? 0)
        self.audioPath = audioPath
        self.transcript = object["transcript_final"]?.string ?? object["transcript_raw"]?.string ?? ""
        self.speakerLabel = object["speaker_label"]?.string ?? "user"
        self.speakerScore = object["speaker_score"]?.double
        self.wakeDetected = object["wake_detected"]?.bool ?? (object["wake_detected"]?.double == 1)
        self.queryCandidate = object["query_candidate"]?.bool ?? (object["query_candidate"]?.double == 1)
        self.queryText = object["query_text"]?.string ?? ""
        self.wasCorrected = object["was_corrected"]?.bool ?? (object["was_corrected"]?.double == 1)
    }
}

struct QueryItem: Identifiable, Hashable {
    let id: String
    let source: String
    let segmentId: String?
    let text: String
    let status: String
    let createdAt: String

    var sortDate: String { createdAt }

    init?(_ value: JSONValue) {
        guard let object = value.object,
              let id = object["id"]?.string,
              let text = object["text"]?.string else { return nil }
        self.id = id
        self.source = object["source"]?.string ?? "voice"
        self.segmentId = object["segment_id"]?.string
        self.text = text
        self.status = object["status"]?.string ?? "pending"
        self.createdAt = object["created_at"]?.string ?? ""
    }
}

enum HistoryItem: Identifiable, Hashable {
    case segment(Segment)
    case query(QueryItem)

    var id: String {
        switch self {
        case .segment(let s): return s.id
        case .query(let q): return q.id
        }
    }

    var sortDate: String {
        switch self {
        case .segment(let s): return s.sortDate
        case .query(let q): return q.sortDate
        }
    }
}

struct ModelStatus: Identifiable, Hashable {
    let key: String
    let ready: Bool
    let path: String
    let errors: [String]
    var id: String { key }

    init?(_ value: JSONValue) {
        guard let object = value.object, let key = object["key"]?.string else { return nil }
        self.key = key
        self.ready = object["ready"]?.bool ?? (object["status"]?.string == "ready")
        self.path = object["path"]?.string ?? ""
        self.errors = object["errors"]?.array?.compactMap(\.string) ?? []
    }
}

struct VoiceSample: Identifiable, Hashable {
    let id: String
    let audioPath: String
    let durationMS: Int
    let score: Double
    let createdAt: String
    let transcriptHint: String?
    let tier: String  // "core" or "cache"
    let isManual: Bool

    init?(_ value: JSONValue) {
        guard let object = value.object,
              let id = object["id"]?.string,
              let audioPath = object["audio_path"]?.string,
              let createdAt = object["created_at"]?.string else { return nil }
        self.id = id
        self.audioPath = audioPath
        self.durationMS = Int(object["duration_ms"]?.double ?? 0)
        self.score = object["score"]?.double ?? 0
        self.createdAt = createdAt
        self.transcriptHint = object["transcript_hint"]?.string
        self.tier = object["tier"]?.string ?? "core"
        self.isManual = object["is_manual"]?.bool ?? (object["is_manual"]?.double == 1)
    }
}

struct LexiconItem: Identifiable, Hashable {
    let id: String
    let word: String
    let weight: Double
    let source: String  // "manual", "auto", "system"
    let useCount: Int
    let createdAt: String

    var sourceLabel: String {
        switch source {
        case "manual": return "手动"
        case "auto": return "自动"
        case "system": return "系统"
        default: return source
        }
    }

    init?(_ value: JSONValue) {
        guard let object = value.object,
              let id = object["id"]?.string,
              let word = object["word"]?.string,
              let source = object["source"]?.string,
              let createdAt = object["created_at"]?.string else { return nil }
        self.id = id
        self.word = word
        self.weight = object["weight"]?.double ?? 2.0
        self.source = source
        self.useCount = Int(object["use_count"]?.double ?? 0)
        self.createdAt = createdAt
    }
}

struct VoiceProfileState: Hashable {
    let sampleCount: Int
    let coreCount: Int
    let cacheCount: Int
    let centroidCount: Int
    let isReady: Bool
    let autoLearnEnabled: Bool
    let lastUpdated: String?

    init(_ value: JSONValue?) {
        guard let object = value?.object else {
            self.sampleCount = 0
            self.coreCount = 0
            self.cacheCount = 0
            self.centroidCount = 0
            self.isReady = false
            self.autoLearnEnabled = true
            self.lastUpdated = nil
            return
        }
        self.sampleCount = Int(object["sample_count"]?.double ?? 0)
        self.coreCount = Int(object["core_count"]?.double ?? 0)
        self.cacheCount = Int(object["cache_count"]?.double ?? 0)
        self.centroidCount = Int(object["centroid_count"]?.double ?? 0)
        self.isReady = object["is_ready"]?.bool ?? (object["exists"]?.bool ?? false)
        self.autoLearnEnabled = object["auto_learn_enabled"]?.bool ?? object["auto_learn"]?.bool ?? true
        self.lastUpdated = object["last_updated"]?.string
    }

    static let empty = VoiceProfileState(nil)
}

enum EngineCommand {
    case status
    case listDevices
    case setInputDevice(index: Int?)
    case startCapture
    case stopCapture
    case setPaused(paused: Bool)
    case submitManualQuery(text: String)
    case listHistory(limit: Int, includeAudioPath: Bool)
    case deleteSegment(id: String)
    case deleteQuery(id: String)
    case clearHistory
    case listModels
    case listQueries(limit: Int)
    case listVoiceSamples
    case deleteVoiceSample(id: String)
    case resetVoiceProfile
    case setAutoLearn(enabled: Bool)
    case playSegment(id: String)
    case shutdown

    func toJSON() -> [String: Any] {
        switch self {
        case .status:
            return ["command": "status"]
        case .listDevices:
            return ["command": "list_devices"]
        case .setInputDevice(let index):
            var params: [String: Any] = [:]
            if let index = index { params["index"] = index }
            return ["command": "set_input_device", "params": params]
        case .startCapture:
            return ["command": "start_capture"]
        case .stopCapture:
            return ["command": "stop_capture"]
        case .setPaused(let paused):
            return ["command": "set_paused", "params": ["paused": paused]]
        case .submitManualQuery(let text):
            return ["command": "submit_manual_query", "params": ["text": text]]
        case .listHistory(let limit, let includeAudioPath):
            return ["command": "list_history", "params": ["limit": limit, "include_audio_path": includeAudioPath]]
        case .deleteSegment(let id):
            return ["command": "delete_segment", "params": ["id": id]]
        case .deleteQuery(let id):
            return ["command": "delete_query", "params": ["id": id]]
        case .clearHistory:
            return ["command": "clear_history"]
        case .listModels:
            return ["command": "list_models"]
        case .listQueries(let limit):
            return ["command": "list_queries", "params": ["limit": limit]]
        case .listVoiceSamples:
            return ["command": "list_voice_samples"]
        case .deleteVoiceSample(let id):
            return ["command": "delete_voice_sample", "params": ["id": id]]
        case .resetVoiceProfile:
            return ["command": "reset_voice_profile"]
        case .setAutoLearn(let enabled):
            return ["command": "set_auto_learn", "params": ["enabled": enabled]]
        case .playSegment(let id):
            return ["command": "play_segment", "params": ["id": id]]
        case .shutdown:
            return ["command": "shutdown"]
        }
    }
}
