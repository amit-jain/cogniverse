import type { ReactActivityMessageRenderer } from '@copilotkit/react-core/v2';

/** The activity type of the notes shown in a conversation; never sent to an agent. */
export const NOTICE_ACTIVITY = 'cogniverse.notice';

export interface Notice {
  tone: 'error' | 'cancelled' | 'warning';
  text: string;
}

function isNotice(value: unknown): value is Notice {
  const notice = value as Partial<Notice> | null;
  return (
    typeof notice?.text === 'string' &&
    (notice.tone === 'error' || notice.tone === 'cancelled' || notice.tone === 'warning')
  );
}

/** A Standard Schema for a notice's content, as CopilotKit renderers take. */
const noticeSchema = {
  '~standard': {
    version: 1 as const,
    vendor: 'cogniverse',
    validate: (value: unknown) =>
      isNotice(value) ? { value } : { issues: [{ message: 'not a cogniverse notice' }] },
  },
};

export function NoticeView({ content }: { content: Notice }) {
  return (
    <p className={`run-notice ${content.tone}`} role={content.tone === 'error' ? 'alert' : 'status'}>
      {content.text}
    </p>
  );
}

export const noticeRenderer: ReactActivityMessageRenderer<Notice> = {
  activityType: NOTICE_ACTIVITY,
  content: noticeSchema,
  render: ({ content }) => <NoticeView content={content} />,
};
