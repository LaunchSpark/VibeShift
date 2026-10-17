/// <reference path="../pb_data/types.d.ts" />
// Initial VibeShift schema. This is the contract shared with the dashboard UI:
// change it only by adding a new migration. All Discord ids are text (snowflakes > 2^53).
migrate((app) => {
  const t = (name) => ({ type: "text", name })
  const d = (name) => ({ type: "date", name })
  const n = (name) => ({ type: "number", name })
  const j = (name) => ({ type: "json", name, maxSize: 5000000 })
  const sel = (name, values) => ({ type: "select", name, values, maxSelect: 1 })

  const defs = [
    {
      name: "guilds",
      fields: [t("guild_id"), t("name"), t("icon_url"), d("history_depth")],
      indexes: ["CREATE UNIQUE INDEX idx_guilds_guild_id ON guilds (guild_id)"],
    },
    {
      name: "channels",
      fields: [t("channel_id"), t("guild_id"), t("name"), t("type"),
               t("newest_fetched_id"), t("oldest_fetched_id")],
      indexes: [
        "CREATE UNIQUE INDEX idx_channels_channel_id ON channels (channel_id)",
        "CREATE INDEX idx_channels_guild_id ON channels (guild_id)",
      ],
    },
    {
      name: "members",
      fields: [t("guild_id"), t("author_id"), t("display_name"), t("avatar_url")],
      indexes: ["CREATE UNIQUE INDEX idx_members_guild_author ON members (guild_id, author_id)"],
    },
    {
      name: "messages",
      fields: [
        t("message_id"), t("guild_id"), t("channel_id"), t("author_id"),
        d("created_at"), d("edited_at"), d("deleted_at"),
        t("reply_to_id"), t("content"), t("day"), t("priority_key"),
        sel("score_status", ["pending", "scored", "failed"]),
        t("emotion"), j("probabilities"), n("confidence"), t("model_id"),
        { type: "bool", name: "rollup_applied" },
      ],
      indexes: [
        "CREATE UNIQUE INDEX idx_messages_message_id ON messages (message_id)",
        "CREATE INDEX idx_messages_status_priority ON messages (score_status, priority_key)",
        "CREATE INDEX idx_messages_guild_channel_created ON messages (guild_id, channel_id, created_at)",
      ],
    },
    {
      name: "daily_rollups",
      fields: [t("guild_id"), t("channel_id"), t("author_id"), t("day"),
               n("count"), j("prob_sums")],
      indexes: [
        "CREATE UNIQUE INDEX idx_rollups_key ON daily_rollups (guild_id, channel_id, author_id, day)",
        "CREATE INDEX idx_rollups_guild_day ON daily_rollups (guild_id, day)",
      ],
    },
    {
      name: "backfill_jobs",
      fields: [
        t("guild_id"), sel("origin", ["downtime", "extend"]), d("target"),
        sel("status", ["pending", "running", "done", "failed"]),
        n("scanned"), t("error"),
        { type: "autodate", name: "created", onCreate: true, onUpdate: false },
        { type: "autodate", name: "updated", onCreate: true, onUpdate: true },
      ],
      indexes: ["CREATE INDEX idx_backfill_status ON backfill_jobs (status)"],
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
  for (const name of ["backfill_jobs", "daily_rollups", "messages", "members", "channels", "guilds"]) {
    app.delete(app.findCollectionByNameOrId(name))
  }
})
