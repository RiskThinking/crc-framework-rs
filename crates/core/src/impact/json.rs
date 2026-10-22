use sha2::{Digest, Sha256};

use super::error::{ContractResult, fail};

#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Json {
    Null,
    Bool(bool),
    Number(f64),
    String(String),
    Array(Vec<Json>),
    Object(Vec<(String, Json)>),
}

impl Json {
    pub(crate) fn object(entries: impl IntoIterator<Item = (&'static str, Json)>) -> Self {
        Self::Object(
            entries
                .into_iter()
                .map(|(key, value)| (key.to_string(), value))
                .collect(),
        )
    }

    pub(crate) fn number(value: f64) -> Self {
        Self::Number(if value == 0.0 { 0.0 } else { value })
    }

    pub(crate) fn number_opt(value: Option<f64>) -> Self {
        value.map(Self::number).unwrap_or(Self::Null)
    }
}

pub(crate) fn pointer(path: &str, key: &str) -> String {
    let token = key.replace('~', "~0").replace('/', "~1");
    format!("{path}/{token}")
}

pub(crate) fn pointer_index(path: &str, index: usize) -> String {
    format!("{path}/{index}")
}

pub(crate) fn parse_json(text: &str) -> ContractResult<Json> {
    if text.starts_with('\u{feff}') {
        return Err(fail(
            "invalid_json",
            "",
            "a leading byte order mark is not allowed",
        ));
    }
    let mut parser = Parser { text, index: 0 };
    parser.skip_whitespace();
    let value = parser.parse_value("")?;
    parser.skip_whitespace();
    if parser.index != parser.text.len() {
        return Err(fail(
            "invalid_json",
            "",
            "trailing data after the JSON value",
        ));
    }
    Ok(value)
}

struct Parser<'a> {
    text: &'a str,
    index: usize,
}

impl<'a> Parser<'a> {
    fn parse_value(&mut self, path: &str) -> ContractResult<Json> {
        self.skip_whitespace();
        match self.peek_char() {
            Some('n') => self.literal("null", path).map(|()| Json::Null),
            Some('t') => self.literal("true", path).map(|()| Json::Bool(true)),
            Some('f') => self.literal("false", path).map(|()| Json::Bool(false)),
            Some('N') => self.literal("NaN", path).and(Err(fail(
                "non_finite_number",
                path,
                "NaN is not a JSON number",
            ))),
            Some('I') => self.literal("Infinity", path).and(Err(fail(
                "non_finite_number",
                path,
                "Infinity is not a JSON number",
            ))),
            Some('"') => self.parse_string(path).map(Json::String),
            Some('[') => self.parse_array(path),
            Some('{') => self.parse_object(path),
            Some('-') if self.remainder().starts_with("-Infinity") => {
                self.literal("-Infinity", path)?;
                Err(fail(
                    "non_finite_number",
                    path,
                    "-Infinity is not a JSON number",
                ))
            }
            Some('-' | '0'..='9') => self.parse_number(path),
            Some(_) => Err(fail("invalid_json", path, "unexpected JSON value")),
            None => Err(fail("invalid_json", path, "unexpected end of JSON")),
        }
    }

    fn parse_array(&mut self, path: &str) -> ContractResult<Json> {
        self.bump();
        let mut items = Vec::new();
        loop {
            self.skip_whitespace();
            if self.consume(']') {
                return Ok(Json::Array(items));
            }
            if !items.is_empty() {
                self.expect(',', path)?;
                self.skip_whitespace();
                if self.consume(']') {
                    return Err(fail("invalid_json", path, "trailing comma in array"));
                }
            }
            items.push(self.parse_value(&pointer_index(path, items.len()))?);
        }
    }

    fn parse_object(&mut self, path: &str) -> ContractResult<Json> {
        self.bump();
        let mut entries = Vec::new();
        loop {
            self.skip_whitespace();
            if self.consume('}') {
                return Ok(Json::Object(entries));
            }
            if !entries.is_empty() {
                self.expect(',', path)?;
                self.skip_whitespace();
                if self.consume('}') {
                    return Err(fail("invalid_json", path, "trailing comma in object"));
                }
            }
            if self.peek_char() != Some('"') {
                return Err(fail("invalid_json", path, "object key must be a string"));
            }
            let key = self.parse_string(path)?;
            let key_path = pointer(path, &key);
            self.skip_whitespace();
            self.expect(':', &key_path)?;
            let value = self.parse_value(&key_path)?;
            if entries.iter().any(|(existing, _)| existing == &key) {
                return Err(fail("duplicate_key", key_path, "duplicate object key"));
            }
            entries.push((key, value));
        }
    }

    fn parse_string(&mut self, path: &str) -> ContractResult<String> {
        self.bump();
        let mut value = String::new();
        loop {
            let character = self.next_char(path)?;
            match character {
                '"' => {
                    if value
                        .chars()
                        .any(|item| (0xD800..=0xDFFF).contains(&(item as u32)))
                    {
                        return Err(fail(
                            "invalid_json",
                            path,
                            "strings must not contain unpaired surrogates",
                        ));
                    }
                    return Ok(value);
                }
                '\\' => value.push(self.parse_escape(path)?),
                character if (character as u32) < 0x20 => {
                    return Err(fail("invalid_json", path, "unescaped control character"));
                }
                character => value.push(character),
            }
        }
    }

    fn parse_escape(&mut self, path: &str) -> ContractResult<char> {
        match self.next_char(path)? {
            '"' => Ok('"'),
            '\\' => Ok('\\'),
            '/' => Ok('/'),
            'b' => Ok('\u{0008}'),
            'f' => Ok('\u{000c}'),
            'n' => Ok('\n'),
            'r' => Ok('\r'),
            't' => Ok('\t'),
            'u' => self.parse_unicode(path),
            _ => Err(fail("invalid_json", path, "invalid string escape")),
        }
    }

    fn parse_unicode(&mut self, path: &str) -> ContractResult<char> {
        let unit = self.hex4(path)?;
        if (0xD800..=0xDBFF).contains(&unit) {
            if self.remainder().starts_with("\\u") {
                self.bump();
                self.bump();
                let low = self.hex4(path)?;
                if (0xDC00..=0xDFFF).contains(&low) {
                    let code = 0x10000 + (((unit as u32) - 0xD800) << 10) + ((low as u32) - 0xDC00);
                    return Ok(char::from_u32(code).expect("paired surrogates form a scalar"));
                }
            }
            return Err(fail("invalid_json", path, "unpaired surrogate"));
        }
        char::from_u32(unit as u32).ok_or_else(|| fail("invalid_json", path, "unpaired surrogate"))
    }

    fn hex4(&mut self, path: &str) -> ContractResult<u16> {
        let start = self.index;
        for _ in 0..4 {
            let character = self.next_char(path)?;
            if !character.is_ascii_hexdigit() {
                return Err(fail("invalid_json", path, "invalid unicode escape"));
            }
        }
        u16::from_str_radix(&self.text[start..self.index], 16)
            .map_err(|_| fail("invalid_json", path, "invalid unicode escape"))
    }

    fn parse_number(&mut self, path: &str) -> ContractResult<Json> {
        let start = self.index;
        if self.consume('-')
            && !self
                .peek_char()
                .is_some_and(|character| character.is_ascii_digit())
        {
            return Err(fail("invalid_json", path, "invalid number"));
        }
        match self.peek_char() {
            Some('0') => {
                self.bump();
                if self
                    .peek_char()
                    .is_some_and(|character| character.is_ascii_digit())
                {
                    return Err(fail("invalid_json", path, "leading zero in number"));
                }
            }
            Some('1'..='9') => while self.consume_if(|character| character.is_ascii_digit()) {},
            _ => return Err(fail("invalid_json", path, "invalid number")),
        }
        if self.consume('.') {
            if !self.consume_if(|character| character.is_ascii_digit()) {
                return Err(fail("invalid_json", path, "invalid fraction"));
            }
            while self.consume_if(|character| character.is_ascii_digit()) {}
        }
        if self.consume('e') || self.consume('E') {
            let _ = self.consume('+') || self.consume('-');
            if !self.consume_if(|character| character.is_ascii_digit()) {
                return Err(fail("invalid_json", path, "invalid exponent"));
            }
            while self.consume_if(|character| character.is_ascii_digit()) {}
        }
        let token = &self.text[start..self.index];
        let value: f64 = token
            .parse()
            .map_err(|_| fail("invalid_json", path, "invalid number"))?;
        if !value.is_finite() {
            return Err(fail(
                "number_overflow",
                path,
                format!("{token} overflows binary64"),
            ));
        }
        Ok(Json::number(value))
    }

    fn literal(&mut self, expected: &str, path: &str) -> ContractResult<()> {
        if self.remainder().starts_with(expected) {
            self.index += expected.len();
            Ok(())
        } else {
            Err(fail("invalid_json", path, "invalid literal"))
        }
    }

    fn expect(&mut self, expected: char, path: &str) -> ContractResult<()> {
        if self.consume(expected) {
            Ok(())
        } else {
            Err(fail("invalid_json", path, format!("expected {expected}")))
        }
    }

    fn consume(&mut self, expected: char) -> bool {
        if self.peek_char() == Some(expected) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn consume_if(&mut self, predicate: impl Fn(char) -> bool) -> bool {
        if self.peek_char().is_some_and(predicate) {
            self.bump();
            true
        } else {
            false
        }
    }

    fn skip_whitespace(&mut self) {
        while self
            .peek_char()
            .is_some_and(|character| matches!(character, ' ' | '\t' | '\n' | '\r'))
        {
            self.bump();
        }
    }

    fn bump(&mut self) -> Option<char> {
        let character = self.peek_char()?;
        self.index += character.len_utf8();
        Some(character)
    }

    fn next_char(&mut self, path: &str) -> ContractResult<char> {
        self.bump()
            .ok_or_else(|| fail("invalid_json", path, "unexpected end of JSON"))
    }

    fn peek_char(&self) -> Option<char> {
        self.text[self.index..].chars().next()
    }

    fn remainder(&self) -> &str {
        &self.text[self.index..]
    }
}

pub(crate) fn canonical_json(value: &Json) -> String {
    fn encode(value: &Json) -> String {
        match value {
            Json::Null => "null".to_string(),
            Json::Bool(true) => "true".to_string(),
            Json::Bool(false) => "false".to_string(),
            Json::Number(number) => {
                if *number == 0.0 {
                    "0.0".to_string()
                } else {
                    serde_json::to_string(number).expect("finite binary64 has a JSON spelling")
                }
            }
            Json::String(text) => {
                serde_json::to_string(text).expect("Rust strings are valid JSON strings")
            }
            Json::Array(items) => {
                let encoded = items.iter().map(encode).collect::<Vec<_>>().join(",");
                format!("[{encoded}]")
            }
            Json::Object(entries) => {
                let mut order = (0..entries.len()).collect::<Vec<_>>();
                order.sort_by(|&left, &right| {
                    entries[left].0.as_bytes().cmp(entries[right].0.as_bytes())
                });
                let encoded = order
                    .into_iter()
                    .map(|index| {
                        let (key, value) = &entries[index];
                        format!(
                            "{}:{}",
                            serde_json::to_string(key).expect("object keys are strings"),
                            encode(value)
                        )
                    })
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{{{encoded}}}")
            }
        }
    }
    encode(value)
}

pub(crate) fn content_hash(value: &Json) -> String {
    let digest = Sha256::digest(canonical_json(value).as_bytes());
    let mut encoded = String::from("sha256:");
    for byte in digest {
        encoded.push_str(&format!("{byte:02x}"));
    }
    encoded
}
