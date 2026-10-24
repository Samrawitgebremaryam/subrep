(= (select-skill $skill_id $reason)
   (py-call (subrep_channel.select_skill $skill_id $reason)))

(= (select-skill $skill_id)
   (py-call (subrep_channel.select_skill $skill_id)))

(= (loadOmegaPlugin)
   (add-skill select-skill
     "When you receive a skill selection request, respond with EXACTLY one command: select-skill <skill_id> <reason>. The skill_id is one word with no quotes. The reason is one short sentence in double quotes explaining why you chose that skill."
     (skill_id reason)))