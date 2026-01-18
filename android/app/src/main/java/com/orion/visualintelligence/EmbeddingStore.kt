package com.orion.visualintelligence

import android.content.Context
import android.util.Log
import org.json.JSONObject

class EmbeddingStore(private val context: Context) {

    val embeddings = mutableMapOf<String, FloatArray>()

    fun load() {
        try {
            val assetManager = context.assets

            val files = assetManager.list("embeddings")
            if (files == null || files.isEmpty()) {
                Log.w("ORION", "⚠️ No embedding files found in assets/embeddings/")
                return
            }

            embeddings.clear()

            for (file in files) {

                if (!file.endsWith(".json")) {
                    Log.w("ORION", "⚠️ Skipping non-JSON file: $file")
                    continue
                }

                val jsonText = assetManager.open("embeddings/$file")
                    .bufferedReader()
                    .use { it.readText() }

                if (jsonText.isBlank()) {
                    Log.w("ORION", "⚠️ Empty embedding file: $file — skipped")
                    continue
                }

                val json = JSONObject(jsonText)

                for (key in json.keys()) {
                    val arr = json.getJSONArray(key)
                    val vector = FloatArray(arr.length())

                    for (i in 0 until arr.length()) {
                        vector[i] = arr.getDouble(i).toFloat()
                    }

                    Similarity.normalize(vector)
                    embeddings[key] = vector
                }
            }

            Log.d("ORION", "✅ Loaded embeddings: ${embeddings.size}")

        } catch (e: Exception) {
            Log.e("ORION", "❌ Error loading embeddings", e)
        }
    }
}
