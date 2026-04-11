import { useMemo, useState } from "react";
import { Stack, IconButton } from "@fluentui/react";
import DOMPurify from "dompurify";

import styles from "./Answer.module.css";

import { RAGChatCompletion } from "../../api/models";
import { parseAnswerToHtml } from "./AnswerParser";
import { AnswerIcon } from "./AnswerIcon";

interface Props {
    answer: RAGChatCompletion;
    isSelected?: boolean;
    isStreaming: boolean;
    onCitationClicked: (filePath: string) => void;
    onThoughtProcessClicked: () => void;
    onSupportingContentClicked: () => void;
}

export const Answer = ({
    answer,
    isSelected,
    isStreaming,
    onCitationClicked,
    onThoughtProcessClicked,
    onSupportingContentClicked
}: Props) => {
    const [isReferencesCollapsed, setIsReferencesCollapsed] = useState(true);
    const messageContent = answer.output_text;
    const parsedAnswer = useMemo(() => parseAnswerToHtml(messageContent, isStreaming, onCitationClicked), [answer]);

    const sanitizedAnswerHtml = DOMPurify.sanitize(parsedAnswer.answerHtml);

    return (
        <Stack className={`${styles.answerContainer} ${isSelected && styles.selected}`} verticalAlign="space-between">
            <Stack.Item>
                <Stack horizontal horizontalAlign="space-between">
                    <AnswerIcon />
                    <div>
                        <IconButton
                            style={{ color: "black" }}
                            iconProps={{ iconName: "Lightbulb" }}
                            title="Show thought process"
                            ariaLabel="Show thought process"
                            onClick={() => onThoughtProcessClicked()}
                            disabled={!answer.context.thoughts?.length}
                        />
                    </div>
                </Stack>
            </Stack.Item>

            <Stack.Item grow>
                <div className={styles.answerText} dangerouslySetInnerHTML={{ __html: sanitizedAnswerHtml }}></div>
            </Stack.Item>

            {!!parsedAnswer.citations.length && (
                <Stack.Item>
                    <Stack horizontal wrap tokens={{ childrenGap: 5 }}>
                        <Stack horizontal verticalAlign="center" tokens={{ childrenGap: 5 }}>
                            <IconButton
                                iconProps={{ iconName: isReferencesCollapsed ? "ChevronDown" : "ChevronUp" }}
                                title={isReferencesCollapsed ? "Expand references" : "Collapse references"}
                                ariaLabel={isReferencesCollapsed ? "Expand references" : "Collapse references"}
                                onClick={() => setIsReferencesCollapsed(!isReferencesCollapsed)}
                            />
                            <span className={styles.citationLearnMore}>References:</span>
                        </Stack>
                    </Stack>
                    {!isReferencesCollapsed && (
                        <ol>
                            {parsedAnswer.citations.map((rowId, ind) => {
                                const citation = answer.context.data_points[rowId];
                                if (!citation) return null;
                                return (
                                    <li key={rowId}>
                                        <h4>{citation.name}</h4>
                                        <p className={styles.referenceMetadata}>Brand: {citation.brand}</p>
                                        <p className={styles.referenceMetadata}>Price: {citation.price}</p>
                                        <p>{citation.description}</p>
                                    </li>
                                );
                            })}
                        </ol>
                    )}
                </Stack.Item>
            )}
        </Stack>
    );
};
