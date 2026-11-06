// Windowed labeling: the imported dataset (label_messages) and the labels people
// make on it (labels). label_messages is written by `pixi run label-import`;
// labels by the dashboard Label page. One labels row per (labeler, window_size, target).
migrate((app) => {
  const t = (name) => ({ type: "text", name })
  const n = (name) => ({ type: "number", name })
  const LABELS = ["joy", "sadness", "anger", "fear", "disgust", "surprise", "neutral"]

  const defs = [
    {
      name: "label_messages",
      fields: [t("message_id"), t("conversation_key"), n("message_number"), t("author"),
               { type: "text", name: "content", max: 0 }, n("queue_position"), n("algorithm_version")],
      indexes: [
        "CREATE UNIQUE INDEX idx_label_messages_message_id ON label_messages (message_id)",
        "CREATE UNIQUE INDEX idx_label_messages_queue ON label_messages (queue_position)",
        "CREATE INDEX idx_label_messages_conv ON label_messages (conversation_key, message_number)",
      ],
    },
    {
      name: "labels",
      fields: [
        t("message_id"), t("labeler"), n("window_size"),
        { type: "select", name: "label", values: LABELS, maxSelect: 1, required: true },
        n("queue_position"), n("algorithm_version"),
        { type: "autodate", name: "created", onCreate: true, onUpdate: false },
        { type: "autodate", name: "updated", onCreate: true, onUpdate: true },
      ],
      indexes: [
        "CREATE UNIQUE INDEX idx_labels_key ON labels (labeler, window_size, message_id)",
        "CREATE INDEX idx_labels_progress ON labels (labeler, window_size, queue_position)",
      ],
    },
  ]

  for (const def of defs) {
    app.save(new Collection({
      type: "base",
      name: def.name,
      listRule: "",
      viewRule: "",
      createRule: "",
      updateRule: "",
      deleteRule: "",
      fields: def.fields,
      indexes: def.indexes,
    }))
  }
}, (app) => {
  for (const name of ["labels", "label_messages"]) {
    app.delete(app.findCollectionByNameOrId(name))
  }
})
