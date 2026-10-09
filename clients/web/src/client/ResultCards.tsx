import { useState, type ReactNode } from 'react';
import { agentLabel } from './api';
import { messageOf } from './ops/common';
import { runtimeJson } from './ops/http';

export interface ResultItem {
  id: string;
  /**
   * The id a relevance rating names: the one the search span records the hit
   * under, and that the triplet miner reads back first.
   */
  ratingId?: string;
  score?: number;
  title?: string;
  snippet?: string;
  /** Segment bounds in seconds, when the hit is a video segment. */
  start?: number;
  end?: number;
  /** The video a segment belongs to, and the backend document it is. */
  videoId?: string;
  documentId?: string;
}

function text(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined;
}

function number(value: unknown): number | undefined {
  return typeof value === 'number' && Number.isFinite(value) ? value : undefined;
}

function record(value: unknown): Record<string, unknown> | undefined {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : undefined;
}

/** One hit in the runtime's public shape for its agent: a video segment, a
 * document, an image or an audio clip. The score is the value the set is
 * ranked by: ``rrf_score`` for an ensemble, ``score`` or ``relevance_score``
 * otherwise. */
function hitOf(entry: unknown): ResultItem[] {
  const hit = record(entry);
  if (!hit) return [];
  const metadata = record(hit.metadata) ?? {};
  const temporal = record(hit.temporal_info) ?? {};
  const id =
    text(hit.id) ??
    text(hit.document_id) ??
    text(hit.image_id) ??
    text(hit.audio_id) ??
    text(hit.video_id) ??
    text(metadata.video_id);
  if (!id) return [];
  return [
    {
      id,
      ratingId:
        text(hit.document_id) ??
        text(hit.documentid) ??
        text(hit.id) ??
        text(hit.source_id) ??
        text(hit.image_id) ??
        text(hit.audio_id) ??
        text(hit.video_id),
      score: number(hit.rrf_score) ?? number(hit.score) ?? number(hit.relevance_score),
      title: text(hit.title) ?? text(metadata.title) ?? text(metadata.video_title),
      snippet:
        text(hit.content_preview) ??
        text(hit.content) ??
        text(hit.text) ??
        text(metadata.segment_description) ??
        text(hit.transcript) ??
        text(hit.description) ??
        text(metadata.audio_transcript) ??
        text(metadata.description),
      start: number(temporal.start_time),
      end: number(temporal.end_time),
      videoId: text(metadata.video_id) ?? text(hit.video_id),
      documentId: text(hit.document_id) ?? text(hit.documentid),
    },
  ];
}

/** The hits of a run's final payload (``state.result.results``). */
export function resultsOf(state: unknown): ResultItem[] {
  const results = record(record(state)?.result)?.results;
  return Array.isArray(results) ? results.flatMap(hitOf) : [];
}

/** The payload of each agent a run's final state holds: the run's own and,
 * for an orchestration, each planned agent's in plan order. */
function payloadsOf(state: unknown): { agent: string; payload: Record<string, unknown> }[] {
  const root = record(state);
  const result = record(root?.result);
  if (!result) return [];
  const agent = text(root?.agent) ?? text(result.agent) ?? '';
  const steps = record(record(result.orchestration_result)?.agent_results) ?? {};
  return [
    { agent, payload: result },
    ...Object.entries(steps).flatMap(([name, payload]) => {
      const step = record(payload);
      return step ? [{ agent: name, payload: step }] : [];
    }),
  ];
}

/** What a search says about itself beside its hits. */
export interface SearchFacts {
  /** The profile searched, or none for an ensemble. */
  profile?: string;
  /** The profiles an ensemble searched. */
  profiles: string[];
  searchMode?: string;
  /** Ensemble legs that did not run, each with its reason. */
  degraded: { profile: string; reason: string }[];
}

export interface ResultGroup {
  agent: string;
  /** The span the agent's search recorded its hits under, when it has one. */
  spanId?: string;
  items: ResultItem[];
  facts: SearchFacts;
  /** The hits as the agent returned them, which an answer agent is grounded in. */
  hits: Record<string, unknown>[];
}

function searchFactsOf(payload: Record<string, unknown>): SearchFacts {
  const profiles = Array.isArray(payload.profiles) ? payload.profiles.flatMap((p) => text(p) ?? []) : [];
  const degraded = Array.isArray(payload.degraded_profiles)
    ? payload.degraded_profiles.flatMap((entry) => {
        const leg = record(entry);
        const profile = text(leg?.profile);
        return profile ? [{ profile, reason: text(leg?.reason) ?? 'no reason given' }] : [];
      })
    : [];
  return { profile: text(payload.profile), profiles, searchMode: text(payload.search_mode), degraded };
}

/** The searches of a run's final state, one per agent that searched, in
 * plan order; a search that found nothing is a group without items. */
export function resultGroupsOf(state: unknown): ResultGroup[] {
  return payloadsOf(state).flatMap(({ agent, payload }) => {
    if (!Array.isArray(payload.results)) return [];
    return [
      {
        agent,
        spanId: text(payload.span_id),
        items: resultsOf({ result: payload }),
        facts: searchFactsOf(payload),
        hits: payload.results.flatMap((hit): Record<string, unknown>[] => {
          const entry = record(hit);
          return entry ? [entry] : [];
        }),
      },
    ];
  });
}

/** The key points an answer agent (the summarizer) gave with its reply. */
export function keyPointsOf(state: unknown): string[] {
  const points = record(record(state)?.result)?.key_points;
  return Array.isArray(points) ? points.flatMap((point) => text(point) ?? []) : [];
}

/** An orchestration's readable account of what its agents did. */
export function orchestrationSummaryOf(state: unknown): string | undefined {
  return text(record(record(record(state)?.result)?.orchestration_result)?.execution_summary);
}

/** An entity extraction's entities and the relationships between them. */
export interface EntityFacts {
  agent: string;
  entities: { text: string; type?: string }[];
  relationships: { subject: string; relation: string; object: string }[];
}

/** The entities of each entity extraction in a run's final state. */
export function entitiesOf(state: unknown): EntityFacts[] {
  return payloadsOf(state).flatMap(({ agent, payload }) => {
    if (!Array.isArray(payload.entities) || !('entity_count' in payload)) return [];
    return [
      {
        agent,
        entities: payload.entities.flatMap((entry) => {
          const entity = record(entry);
          const name = text(entity?.text);
          return name ? [{ text: name, type: text(entity?.type) }] : [];
        }),
        relationships: (Array.isArray(payload.relationships) ? payload.relationships : []).flatMap((entry) => {
          const relation = record(entry);
          const subject = text(relation?.subject);
          const verb = text(relation?.relation);
          const object = text(relation?.object);
          return subject && verb && object ? [{ subject, relation: verb, object }] : [];
        }),
      },
    ];
  });
}

/** A query enhancement: the query as asked, the query searched and why. */
export interface EnhancementFacts {
  agent: string;
  original: string;
  enhanced: string;
  expansionTerms: string[];
  synonyms: string[];
  variants: string[];
  path?: string;
  reasoning?: string;
}

const strings = (value: unknown): string[] => (Array.isArray(value) ? value.flatMap((entry) => text(entry) ?? []) : []);

/** Each query enhancement in a run's final state. */
export function enhancementsOf(state: unknown): EnhancementFacts[] {
  return payloadsOf(state).flatMap(({ agent, payload }) => {
    const original = text(payload.original_query);
    const enhanced = text(payload.enhanced_query);
    if (original === undefined || enhanced === undefined) return [];
    return [
      {
        agent,
        original,
        enhanced,
        expansionTerms: strings(payload.expansion_terms),
        synonyms: strings(payload.synonyms),
        variants: strings(payload.query_variants),
        path: text(payload.path_used),
        reasoning: text(payload.reasoning),
      },
    ];
  });
}

/** A profile selection: the profile chosen for a query, and the runners-up. */
export interface ProfileChoice {
  agent: string;
  profile: string;
  confidence?: number;
  intent?: string;
  modality?: string;
  complexity?: string;
  reasoning?: string;
  alternatives: { profile: string; score?: number; reasoning?: string }[];
}

/** Each profile selection in a run's final state. */
export function profileSelectionsOf(state: unknown): ProfileChoice[] {
  return payloadsOf(state).flatMap(({ agent, payload }) => {
    const profile = text(payload.selected_profile);
    if (!profile) return [];
    return [
      {
        agent,
        profile,
        confidence: number(payload.confidence),
        intent: text(payload.query_intent),
        modality: text(payload.modality),
        complexity: text(payload.complexity),
        reasoning: text(payload.reasoning),
        alternatives: (Array.isArray(payload.alternatives) ? payload.alternatives : []).flatMap((entry) => {
          const candidate = record(entry);
          const name = text(candidate?.profile_name);
          return name ? [{ profile: name, score: number(candidate?.score), reasoning: text(candidate?.reasoning) }] : [];
        }),
      },
    ];
  });
}

/** "video_search" -> "video search". */
const words = (value: string) => value.replace(/_/g, ' ');

export interface CodingResult {
  agent: string;
  summary?: string;
  files: { path: string; content: string; change?: string }[];
  runs: { command?: string; exitCode?: number; stdout: string; stderr: string }[];
}

/** The code a run wrote and the output of running it, from the coding
 * agent's output, whether it arrives as the dispatcher's envelope
 * (``result.result``), as the streamed output itself, or as one step of an
 * orchestration. */
export function codingOf(state: unknown): CodingResult[] {
  return payloadsOf(state).flatMap(({ agent, payload }) => {
    const output = Array.isArray(payload.code_changes) ? payload : record(payload.result);
    if (!output || !Array.isArray(output.code_changes)) return [];
    const runs = Array.isArray(output.execution_results) ? output.execution_results : [];
    return [
      {
        agent,
        summary: text(output.summary),
        files: output.code_changes.flatMap((entry) => {
          const change = record(entry);
          const path = text(change?.file_path);
          return change && path
            ? [{ path, content: typeof change.content === 'string' ? change.content : '', change: text(change.change_type) }]
            : [];
        }),
        runs: runs.flatMap((entry) => {
          const run = record(entry);
          return run
            ? [
                {
                  command: text(run.command),
                  exitCode: number(run.exit_code),
                  stdout: typeof run.stdout === 'string' ? run.stdout : '',
                  stderr: typeof run.stderr === 'string' ? run.stderr : '',
                },
              ]
            : [];
        }),
      },
    ];
  });
}

/** The search's telemetry span id in a run's final payload, when it has one. */
export function searchSpanOf(state: unknown): string | undefined {
  return text(record(record(state)?.result)?.span_id);
}

/** The labels a reviewer rates a hit with, as the runtime stores them. */
export const RELEVANCE_LABELS = ['Highly Relevant', 'Somewhat Relevant', 'Not Relevant'] as const;

/** 75.4 -> "1:15". */
export function clock(seconds: number): string {
  const whole = Math.floor(seconds);
  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, '0')}`;
}

/** The tenant a run's final state was produced for. */
export function resultTenantOf(state: unknown): string | undefined {
  return text(record(state)?.tenant_id);
}

/** A relevance rating that was stored. */
export interface Rating {
  spanId: string;
  resultId: string;
  relevance: string;
  score: number;
}

/** What the panel knows about the run beside its final state. */
export interface RunFacts {
  /** The question the run answered. */
  query?: string;
  /** How long the run took, start to finish. */
  latencyMs?: number;
  /** Hits scoring below this are hidden. */
  minScore?: number;
}

/** "Found 2 results for 'cats'" or "No results for 'cats'". */
export function foundLine(count: number, query?: string): string {
  const what = query ? ` for '${query}'` : '';
  if (count === 0) return `No results${what}.`;
  return `Found ${count} result${count === 1 ? '' : 's'}${what}.`;
}

/** The hits, code, key points, orchestration account, entities, query
 * enhancement and profile selection of a run's final state, beside the chat; ``children`` follow them. A run's results are
 * shown only while ``tenant`` is the tenant the run was produced for. */
export function ResultPanel({
  state,
  tenant,
  run = {},
  onRated,
  children,
}: {
  state: unknown;
  tenant: string;
  run?: RunFacts;
  onRated?: (rating: Rating) => void;
  children?: ReactNode;
}) {
  const groups = resultGroupsOf(state);
  const coding = codingOf(state);
  const keyPoints = keyPointsOf(state);
  const orchestration = orchestrationSummaryOf(state);
  const entities = entitiesOf(state);
  const enhancements = enhancementsOf(state);
  const profileChoices = profileSelectionsOf(state);
  const enrichments = entities.length + enhancements.length + profileChoices.length;
  const owner = resultTenantOf(state);
  if ((groups.length || coding.length || keyPoints.length || orchestration || enrichments) && owner !== tenant)
    return (
      <aside className="results" aria-label="Results">
        <p className="alert error" role="alert">
          These results belong to tenant {owner ?? 'unknown'}, not {tenant || 'the active tenant'}; they are not
          shown.
        </p>
      </aside>
    );
  const labelled = groups.length > 1;
  const empty = !groups.length && !coding.length && !keyPoints.length && !orchestration && !enrichments;
  return (
    <aside className="results" aria-label="Results">
      {empty && !children && <p className="muted">Results of a run appear here.</p>}
      {orchestration && (
        <section className="orchestration-summary" aria-label="Orchestration summary">
          <h2 className="results-heading">Orchestration</h2>
          <p>{orchestration}</p>
        </section>
      )}
      {keyPoints.length > 0 && (
        <section className="key-points" aria-label="Key points">
          <h2 className="results-heading">Key points</h2>
          <ul>
            {keyPoints.map((point, index) => (
              <li key={index}>{point}</li>
            ))}
          </ul>
        </section>
      )}
      {entities.map((facts) => (
        <EntitiesPanel key={facts.agent} facts={facts} />
      ))}
      {enhancements.map((facts) => (
        <EnhancementPanel key={facts.agent} facts={facts} />
      ))}
      {profileChoices.map((choice) => (
        <ProfileChoicePanel key={choice.agent} choice={choice} />
      ))}
      {coding.map((code) => (
        <CodePanel key={code.agent} code={code} />
      ))}
      {groups.map((group) => (
        <SearchGroup
          key={`${group.agent}-${group.spanId ?? ''}`}
          group={group}
          labelled={labelled}
          run={run}
          tenant={tenant}
          onRated={onRated}
        />
      ))}
      {children}
    </aside>
  );
}

function SearchGroup({
  group,
  labelled,
  run,
  tenant,
  onRated,
}: {
  group: ResultGroup;
  labelled: boolean;
  run: RunFacts;
  tenant: string;
  onRated?: (rating: Rating) => void;
}) {
  const label = agentLabel(group.agent);
  const { facts } = group;
  const minScore = run.minScore ?? 0;
  const shown = group.items.filter((item) => item.score === undefined || item.score >= minScore);
  const profile = facts.profile ?? (facts.profiles.length ? facts.profiles.join(', ') : undefined);
  return (
    <section className="search-group" aria-label={`Search by ${label}`}>
      {labelled && <h2 className="result-group">{label}</h2>}
      <p className="result-found">{foundLine(group.items.length, run.query)}</p>
      <dl className="result-metrics">
        <div>
          <dt>Results</dt>
          <dd>{group.items.length}</dd>
        </div>
        {run.latencyMs !== undefined && (
          <div>
            <dt>Latency</dt>
            <dd>{`${Math.round(run.latencyMs)} ms`}</dd>
          </div>
        )}
        {profile && (
          <div>
            <dt>Profile</dt>
            <dd>{profile}</dd>
          </div>
        )}
        {facts.searchMode && (
          <div>
            <dt>Search mode</dt>
            <dd>{facts.searchMode}</dd>
          </div>
        )}
      </dl>
      {facts.degraded.map((leg) => (
        <p key={leg.profile} className="alert warning" role="alert">
          {`Partial results: ${leg.profile} did not run (${leg.reason}).`}
        </p>
      ))}
      {shown.length < group.items.length && (
        <p className="muted">{`Showing ${shown.length} of ${group.items.length}; the rest score below ${minScore}.`}</p>
      )}
      <ResultCards results={shown} spanId={group.spanId} tenant={tenant} onRated={onRated} />
    </section>
  );
}

function Fact({ term, children }: { term: string; children: ReactNode }) {
  return (
    <div>
      <dt>{term}</dt>
      <dd>{children}</dd>
    </div>
  );
}

function EntitiesPanel({ facts }: { facts: EntityFacts }) {
  return (
    <section className="entities" aria-label="Entities">
      <h2 className="results-heading">Entities</h2>
      {facts.entities.length === 0 ? (
        <p className="muted">No entities found.</p>
      ) : (
        <ul>
          {facts.entities.map((entity, index) => (
            <li key={index}>
              {entity.text}
              {entity.type && <span className="entity-type">{` ${entity.type.toLowerCase()}`}</span>}
            </li>
          ))}
        </ul>
      )}
      {facts.relationships.length > 0 && (
        <>
          <h3 className="results-subheading">Relationships</h3>
          <ul>
            {facts.relationships.map((relation, index) => (
              <li key={index}>{`${relation.subject} → ${relation.relation} → ${relation.object}`}</li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function EnhancementPanel({ facts }: { facts: EnhancementFacts }) {
  return (
    <section className="query-enhancement" aria-label="Query enhancement">
      <h2 className="results-heading">Query enhancement</h2>
      <dl className="enrichment-facts">
        <Fact term="Asked">{facts.original}</Fact>
        <Fact term="Searched">{facts.enhanced === facts.original ? 'Unchanged' : facts.enhanced}</Fact>
        {facts.expansionTerms.length > 0 && <Fact term="Expansion terms">{facts.expansionTerms.join(', ')}</Fact>}
        {facts.synonyms.length > 0 && <Fact term="Synonyms">{facts.synonyms.join(', ')}</Fact>}
        {facts.variants.length > 0 && <Fact term="Variants">{facts.variants.join(' · ')}</Fact>}
        {facts.path && <Fact term="Path">{facts.path === 'lm' ? 'Language model' : words(facts.path)}</Fact>}
      </dl>
      {facts.reasoning && <p className="enrichment-reasoning">{facts.reasoning}</p>}
    </section>
  );
}

function ProfileChoicePanel({ choice }: { choice: ProfileChoice }) {
  return (
    <section className="profile-selection" aria-label="Profile selection">
      <h2 className="results-heading">Profile selection</h2>
      <dl className="enrichment-facts">
        <Fact term="Profile">{choice.profile}</Fact>
        {choice.confidence !== undefined && <Fact term="Confidence">{choice.confidence.toFixed(2)}</Fact>}
        {choice.intent && <Fact term="Intent">{words(choice.intent)}</Fact>}
        {choice.modality && <Fact term="Modality">{choice.modality}</Fact>}
        {choice.complexity && <Fact term="Complexity">{choice.complexity}</Fact>}
      </dl>
      {choice.reasoning && <p className="enrichment-reasoning">{choice.reasoning}</p>}
      {choice.alternatives.length > 0 && (
        <>
          <h3 className="results-subheading">Alternatives</h3>
          <ul>
            {choice.alternatives.map((alternative) => (
              <li key={alternative.profile}>
                {[
                  alternative.score === undefined
                    ? alternative.profile
                    : `${alternative.profile} (${alternative.score.toFixed(2)})`,
                  alternative.reasoning,
                ]
                  .filter(Boolean)
                  .join(': ')}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function CodePanel({ code }: { code: CodingResult }) {
  return (
    <section className="code-result" aria-label={`Code from ${agentLabel(code.agent)}`}>
      {code.summary && <p className="code-summary">{code.summary}</p>}
      {code.files.map((file) => (
        <figure key={file.path} className="code-file">
          <figcaption>{file.change ? `${file.path} (${file.change})` : file.path}</figcaption>
          <pre>
            <code>{file.content}</code>
          </pre>
        </figure>
      ))}
      {code.runs.map((run, index) => (
        <figure key={index} className="code-run">
          <figcaption>
            {[run.command, run.exitCode === undefined ? undefined : `exit code ${run.exitCode}`]
              .filter(Boolean)
              .join(' — ')}
          </figcaption>
          {run.stdout && <pre aria-label="Output">{run.stdout}</pre>}
          {run.stderr && (
            <pre className="code-stderr" aria-label="Errors">
              {run.stderr}
            </pre>
          )}
        </figure>
      ))}
    </section>
  );
}

export function ResultCards({
  results,
  spanId,
  tenant,
  onRated,
}: {
  results: ResultItem[];
  spanId?: string;
  tenant: string;
  onRated?: (rating: Rating) => void;
}) {
  return (
    <ol className="result-list">
      {results.map((item, index) => (
        <li key={`${item.id}-${index}`} className="result-card">
          <div className="result-head">
            <span className="result-title">{item.title ?? item.id}</span>
            {item.score !== undefined && (
              <span className="result-score">{item.score.toFixed(3)}</span>
            )}
          </div>
          {(item.title || item.start !== undefined) && (
            <div className="result-id">
              {item.title && item.id}
              {item.start !== undefined && item.end !== undefined && (
                <span className="result-time">{` ${clock(item.start)}–${clock(item.end)}`}</span>
              )}
            </div>
          )}
          {(item.videoId || item.documentId) && (
            <div className="result-id">
              {[item.videoId && `Video ${item.videoId}`, item.documentId && `Document ${item.documentId}`]
                .filter(Boolean)
                .join(' · ')}
            </div>
          )}
          {item.snippet && <p className="result-snippet">{item.snippet}</p>}
          {spanId && item.ratingId && (
            <Relevance spanId={spanId} resultId={item.ratingId} tenant={tenant} onRated={onRated} />
          )}
        </li>
      ))}
    </ol>
  );
}

function Relevance({
  spanId,
  resultId,
  tenant,
  onRated,
}: {
  spanId: string;
  resultId: string;
  tenant: string;
  onRated?: (rating: Rating) => void;
}) {
  const [rated, setRated] = useState<string>();
  const [pending, setPending] = useState<string>();
  const [error, setError] = useState('');
  const rate = async (relevance: string) => {
    setPending(relevance);
    setError('');
    try {
      const stored = await runtimeJson<{ relevance: string; score: number }>('/ag-ui/results/relevance', {
        method: 'POST',
        body: { span_id: spanId, result_id: resultId, relevance },
        tenant,
      });
      setRated(stored.relevance);
      onRated?.({ spanId, resultId, relevance: stored.relevance, score: stored.score });
    } catch (e) {
      setError(messageOf(e));
    } finally {
      setPending(undefined);
    }
  };
  return (
    <div className="relevance" role="group" aria-label={`Relevance of ${resultId}`}>
      {RELEVANCE_LABELS.map((label) => (
        <button
          key={label}
          type="button"
          aria-pressed={rated === label}
          disabled={pending !== undefined}
          onClick={() => rate(label)}
        >
          {pending === label ? 'Saving…' : label}
        </button>
      ))}
      {error && (
        <p className="relevance-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
