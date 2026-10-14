package gateway

import "strings"

// A model asked for JSON through a translated format (a Gemini model's
// responseJsonSchema, a Chat relay's response_format, a format told in the
// system prompt) may still wrap its answer in a Markdown code fence:
// ```json\n{…}\n```. A client that asked for structured output parses the
// text as JSON as it is, so the fenced answer fails it: Codex's memory
// extraction on such a model failed every time with "expected value at
// line 1 column 1". fenceEncoder and unfencedResult hand the client the JSON
// inside the fence; any other answer passes as it came.

// unfence is said with a code fence around the whole of it taken off, and
// whether there was one: an opening ``` (with or without a language tag)
// on the first line and a closing ``` that ends the text.
func unfence(said string) (string, bool) {
	t := strings.TrimSpace(said)
	if len(t) < 6 || !strings.HasPrefix(t, "```") || !strings.HasSuffix(t, "```") {
		return said, false
	}
	nl := strings.IndexByte(t, '\n')
	if nl < 0 {
		return said, false
	}
	if tag := strings.TrimSpace(t[3:nl]); strings.ContainsAny(tag, " {[\"`") {
		return said, false
	}
	inner := strings.TrimSpace(t[nl+1 : len(t)-3])
	if inner == "" || strings.Contains(inner, "```") {
		return said, false
	}
	return inner, true
}

// unfencedResult is res with its answer's fence taken off, for a client that
// asked for structured output.
func unfencedResult(res Result, r *Request) Result {
	if !r.wantsJSON() {
		return res
	}
	var parts []Part
	for i, p := range res.Parts {
		if p.Kind != Text {
			continue
		}
		if t, ok := unfence(p.Text); ok {
			if parts == nil {
				parts = append([]Part(nil), res.Parts...)
			}
			parts[i].Text = t
		}
	}
	if parts != nil {
		res.Parts = parts
	}
	return res
}

// fenceEncoder holds a structured-output reply's text while it could still
// be a fenced answer: once it begins with anything but ``` it passes on as
// it comes; a reply that began with a fence is handed on whole, unfenced,
// when its text ends.
type fenceEncoder struct {
	streamEncoder
	held    strings.Builder
	holding bool // the text so far could be a fence's start
	past    bool // the text began otherwise; nothing is held now
}

func (e *fenceEncoder) event(ev Event) {
	if ev.Kind != KText {
		if ev.Kind != KStart && ev.Kind != KUsage {
			e.flush()
		}
		e.streamEncoder.event(ev)
		return
	}
	if e.past {
		e.streamEncoder.event(ev)
		return
	}
	e.held.WriteString(ev.Text)
	lead := strings.TrimLeft(e.held.String(), " \t\r\n")
	if lead == "" || strings.HasPrefix("```", lead) || strings.HasPrefix(lead, "```") {
		e.holding = true
		return
	}
	e.past, e.holding = true, false
	ev.Text = e.held.String()
	e.held.Reset()
	e.streamEncoder.event(ev)
}

// flush hands on the text held, unfenced when it is a fenced answer.
func (e *fenceEncoder) flush() {
	if !e.holding {
		return
	}
	e.holding, e.past = false, true
	t := e.held.String()
	e.held.Reset()
	if u, ok := unfence(t); ok {
		t = u
	}
	if t != "" {
		e.streamEncoder.event(Event{Kind: KText, Text: t})
	}
}

func (e *fenceEncoder) finish() {
	e.flush()
	e.streamEncoder.finish()
}
