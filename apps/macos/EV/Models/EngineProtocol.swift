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
    let queryText: String?

    init?(_ object: [String: JSONValue]) {
        guard let id = object["id"]?.string,
              let startedAt = object["started_at"]?.string,
              let audioPath = object["audio_path"]?.string else { return nil }
        self.id = id
        self.startedAt = startedAt
        self.durationMS = Int(object["duration_ms"]?.double ?? 0)
        self.audioPath = audioPath
        self.transcript = object["transcript_final"]?.string ?? object["transcript_raw"]?.string ?? ""
        self.speakerLabel = object["speaker_label"]?.string ?? "unknown"
        self.speakerScore = object["speaker_score"]?.double
        self.wakeDetected = object["wake_detected"]?.bool ?? (object["wake_detected"]?.double == 1)
        self.queryCandidate = object["query_candidate"]?.bool ?? (object["query_candidate"]?.double == 1)
        self.queryText = object["query_text"]?.string
    }
}

struct QueryItem: Identifiable, Hashable {
    let id: String
    let source: String
    let text: String
    let status: String
    let createdAt: String

    init?(_ value: JSONValue) {
        guard let object = value.object,
              let id = object["id"]?.string,
              let text = object["text"]?.string else { return nil }
        self.id = id
        self.source = object["source"]?.string ?? "voice"
        self.text = text
        self.status = object["status"]?.string ?? "pending"
        self.createdAt = object["created_at"]?.string ?? ""
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
