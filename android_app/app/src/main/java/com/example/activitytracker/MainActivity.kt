package com.example.activitytracker

import android.Manifest
import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.content.ContentValues
import android.content.Context
import android.content.pm.PackageManager
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.media.AudioManager
import android.media.ToneGenerator
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.os.SystemClock
import android.provider.MediaStore
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxHeight
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.core.content.ContextCompat
import kotlinx.coroutines.delay
import org.tensorflow.lite.Interpreter
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.time.LocalDate
import java.time.format.TextStyle
import java.util.Locale
import kotlin.math.sqrt

// ---- settings that must match the training notebook ----
private const val WINDOW = 100          // 100 rows = 5 seconds at 20 Hz
private const val N_CH = 6              // ax, ay, az, gx, gy, gz
private val LABELS = listOf("Walking", "Jogging", "Sitting", "Standing")   // Walking also covers Stairs
private val MEAN = floatArrayOf(1.0253f, -1.7554f, -0.2430f, -0.0112f, -0.0079f, -0.0020f)
private val STD  = floatArrayOf(6.4385f, 8.1821f, 4.6221f, 1.2862f, 1.5038f, 1.0705f)

// ---- step detector settings (tuned values) ----
private const val STEP_THRESHOLD = 2.0f
private const val MIN_STEP_MS = 350L

// ---- recording settings ----
private const val POSITION = "pocket"
private const val PREP_SECONDS = 10
private const val RECORD_SECONDS = 60

// ---- look ----
private val Teal = Color(0xFF2E7D6B)
private val Orange = Color(0xFFF28C38)
private val Blue = Color(0xFF4A78C2)
private val Grey = Color(0xFF8A9690)
private val Bg = Color(0xFFF4F7F5)
private val Track = Color(0xFFE3EBE7)
private val Ink = Color(0xFF1F2A26)
private val ACT_COLORS = listOf(Teal, Orange, Blue, Grey)       // Walking, Jogging, Sitting, Standing
private val REMINDER_OPTIONS = listOf(1, 5, 15, 30, 45, 60, 90, 120)   // minutes (1 is handy for a demo)
private const val CHANNEL = "reminders"

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)   // sensors stop if the screen turns off
        enableEdgeToEdge()
        setContent {
            MaterialTheme(
                colorScheme = lightColorScheme(
                    primary = Teal, onPrimary = Color.White, secondary = Orange,
                    background = Bg, surface = Color.White, onSurface = Ink, onBackground = Ink
                )
            ) {
                Scaffold(modifier = Modifier.fillMaxSize(), containerColor = Bg) { innerPadding ->
                    ActivityScreen(Modifier.padding(innerPadding))
                }
            }
        }
    }
}

private fun loadModel(context: Context): Interpreter {
    val bytes = context.assets.open("cnn4_float16.tflite").use { it.readBytes() }
    val buf = ByteBuffer.allocateDirect(bytes.size).order(ByteOrder.nativeOrder())
    buf.put(bytes)
    buf.rewind()
    return Interpreter(buf)
}

private fun saveCsv(context: Context, fileName: String, text: String): String {
    return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
        val values = ContentValues().apply {
            put(MediaStore.MediaColumns.DISPLAY_NAME, fileName)
            put(MediaStore.MediaColumns.MIME_TYPE, "text/csv")
            put(MediaStore.MediaColumns.RELATIVE_PATH, Environment.DIRECTORY_DOWNLOADS + "/ActivityData")
        }
        val uri = context.contentResolver.insert(MediaStore.Downloads.EXTERNAL_CONTENT_URI, values)!!
        context.contentResolver.openOutputStream(uri)!!.use { it.write(text.toByteArray()) }
        "Downloads/ActivityData/$fileName"
    } else {
        File(context.getExternalFilesDir(null)!!, fileName).writeText(text)
        "app files/$fileName"
    }
}

// shows a notification (needs the notification permission on Android 13+)
private fun sendReminder(context: Context, text: String) {
    val nm = context.getSystemService(Context.NOTIFICATION_SERVICE) as NotificationManager
    nm.createNotificationChannel(NotificationChannel(CHANNEL, "Reminders", NotificationManager.IMPORTANCE_DEFAULT))
    val n = Notification.Builder(context, CHANNEL)
        .setSmallIcon(android.R.drawable.ic_dialog_info)
        .setContentTitle("Time to move")
        .setContentText(text)
        .setAutoCancel(true)
        .build()
    try { nm.notify(1, n) } catch (e: SecurityException) { }
}

private fun fmt(s: Long) = "${s / 60} min ${s % 60} s"

// ---------- small building blocks ----------
@Composable
fun AppCard(content: @Composable () -> Unit) {
    Card(
        modifier = Modifier.fillMaxWidth(),
        shape = RoundedCornerShape(20.dp),
        colors = CardDefaults.cardColors(containerColor = Color.White),
        elevation = CardDefaults.cardElevation(defaultElevation = 2.dp)
    ) {
        Column(Modifier.padding(16.dp), verticalArrangement = Arrangement.spacedBy(8.dp)) { content() }
    }
}

@Composable
fun Bar(frac: Float, color: Color) {           // a simple rounded progress bar
    Box(Modifier.fillMaxWidth().height(10.dp).clip(RoundedCornerShape(5.dp)).background(Track)) {
        if (frac > 0f) Box(Modifier.fillMaxWidth(frac.coerceIn(0f, 1f)).fillMaxHeight().background(color))
    }
}

@Composable
fun Ring(frac: Float, color: Color, modifier: Modifier, content: @Composable () -> Unit) {   // circular progress
    Box(modifier, contentAlignment = Alignment.Center) {
        Canvas(Modifier.fillMaxSize()) {
            val w = 16.dp.toPx()
            val s = Size(size.width - w, size.height - w)
            val tl = Offset(w / 2, w / 2)
            drawArc(color = Track, startAngle = -90f, sweepAngle = 360f, useCenter = false, topLeft = tl, size = s, style = Stroke(w))
            drawArc(color = color, startAngle = -90f, sweepAngle = 360f * frac.coerceIn(0f, 1f), useCenter = false,
                topLeft = tl, size = s, style = Stroke(w, cap = StrokeCap.Round))
        }
        content()
    }
}

@Composable
fun BarChart(values: List<Float>, labels: List<String>, color: Color, goal: Float) {
    val maxV = maxOf(values.maxOrNull() ?: 0f, goal, 1f)
    Row(Modifier.fillMaxWidth().height(150.dp), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.Bottom) {
        values.forEachIndexed { i, v ->
            Column(Modifier.weight(1f).fillMaxHeight(), horizontalAlignment = Alignment.CenterHorizontally, verticalArrangement = Arrangement.Bottom) {
                Text(if (v > 0f) v.toInt().toString() else "", fontSize = 10.sp, color = Grey)
                Box(
                    Modifier.fillMaxWidth().height((100f * v / maxV).coerceAtLeast(2f).dp)
                        .clip(RoundedCornerShape(topStart = 6.dp, topEnd = 6.dp))
                        .background(if (v >= goal) color else color.copy(alpha = 0.45f))
                )
                Text(labels[i], fontSize = 11.sp, color = Grey)
            }
        }
    }
}

@Composable
fun ActivityScreen(modifier: Modifier = Modifier) {
    val context = LocalContext.current
    val prefs = remember { context.getSharedPreferences("tracker", Context.MODE_PRIVATE) }

    var tab by remember { mutableStateOf(0) }
    val tabs = listOf("Today", "Week", "Goals", "Data")

    // ---- live values ----
    var rawLabel by remember { mutableStateOf("-") }
    var smoothLabel by remember { mutableStateOf("-") }
    var probs by remember { mutableStateOf(FloatArray(4)) }
    var filled by remember { mutableStateOf(0) }
    var rowsPerSec by remember { mutableStateOf(0) }
    var sensorMissing by remember { mutableStateOf(false) }
    var noStepSensor by remember { mutableStateOf(false) }
    var todaySecs by remember { mutableStateOf(LongArray(4) { prefs.getLong("t_${LocalDate.now()}_${LABELS[it]}", 0L) }) }
    val hasBarometer = remember {
        (context.getSystemService(Context.SENSOR_SERVICE) as SensorManager).getDefaultSensor(Sensor.TYPE_PRESSURE) != null
    }

    // ---- steps ----
    var builtInSteps by remember { mutableStateOf(prefs.getInt("raw_${LocalDate.now()}", 0)) }
    var ownSteps by remember { mutableStateOf(prefs.getInt("own_${LocalDate.now()}", 0)) }
    var ownIgnored by remember { mutableStateOf(prefs.getInt("ownign_${LocalDate.now()}", 0)) }
    val moving = remember { BooleanArray(1) }
    val pend = remember { IntArray(1) }
    val pendT = remember { LongArray(1) }
    val base = remember { floatArrayOf(9.8f) }
    val dev1 = remember { FloatArray(1) }
    val dev2 = remember { FloatArray(1) }
    val lastStepMs = remember { LongArray(1) }

    // ---- goals and reminder ----
    var goalSteps by remember { mutableStateOf(prefs.getInt("goal_steps", 6000)) }
    var goalActive by remember { mutableStateOf(prefs.getInt("goal_active", 30)) }
    var reminderMin by remember { mutableStateOf(prefs.getInt("reminder_min", 60)) }
    var stillSecs by remember { mutableStateOf(0) }
    val reminded = remember { BooleanArray(1) }

    fun commit(n: Int) {
        val d = LocalDate.now().toString()
        val v = prefs.getInt("own_$d", 0) + n
        prefs.edit().putInt("own_$d", v).apply()
        ownSteps = v
    }
    fun ignore(n: Int) {
        val d = LocalDate.now().toString()
        val v = prefs.getInt("ownign_$d", 0) + n
        prefs.edit().putInt("ownign_$d", v).apply()
        ownIgnored = v
    }
    fun resetSteps() {
        val d = LocalDate.now().toString()
        prefs.edit().putInt("own_$d", 0).putInt("ownign_$d", 0).putInt("raw_$d", 0).apply()
        ownSteps = 0; ownIgnored = 0; builtInSteps = 0; pend[0] = 0
    }

    // ---- permissions (step counter, notifications) ----
    var stepPerm by remember {
        mutableStateOf(
            Build.VERSION.SDK_INT < Build.VERSION_CODES.Q ||
                    ContextCompat.checkSelfPermission(context, Manifest.permission.ACTIVITY_RECOGNITION) == PackageManager.PERMISSION_GRANTED
        )
    }
    val permLauncher = rememberLauncherForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) {
        stepPerm = Build.VERSION.SDK_INT < Build.VERSION_CODES.Q ||
                ContextCompat.checkSelfPermission(context, Manifest.permission.ACTIVITY_RECOGNITION) == PackageManager.PERMISSION_GRANTED
    }
    LaunchedEffect(Unit) {
        val need = ArrayList<String>()
        if (!stepPerm) need.add(Manifest.permission.ACTIVITY_RECOGNITION)
        if (Build.VERSION.SDK_INT >= 33 &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.POST_NOTIFICATIONS) != PackageManager.PERMISSION_GRANTED)
            need.add(Manifest.permission.POST_NOTIFICATIONS)
        if (need.isNotEmpty()) permLauncher.launch(need.toTypedArray())
    }

    // ---- recording state ----
    var selected by remember { mutableStateOf(0) }
    var phase by remember { mutableStateOf(0) }
    var secondsLeft by remember { mutableStateOf(0) }
    var startToken by remember { mutableStateOf(0) }
    var message by remember { mutableStateOf("") }
    val recRows = remember { ArrayList<FloatArray>() }
    val tone = remember { ToneGenerator(AudioManager.STREAM_MUSIC, 100) }

    val loaded = remember { runCatching { loadModel(context) } }
    val interpreter = loaded.getOrNull()
    val loadError = loaded.exceptionOrNull()?.message

    // ---- helpers for the 20 Hz stream ----
    val accSum = remember { FloatArray(3) }
    val gyroSum = remember { FloatArray(3) }
    val counts = remember { IntArray(2) }
    val lastAcc = remember { FloatArray(3) }
    val lastGyro = remember { FloatArray(3) }
    val seen = remember { BooleanArray(2) }
    val rows = remember { ArrayDeque<FloatArray>() }
    val recent = remember { ArrayDeque<Int>() }

    // 1) motion sensors
    DisposableEffect(Unit) {
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val acc = sm.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val gyro = sm.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        val listener = object : SensorEventListener {
            override fun onSensorChanged(e: SensorEvent) {
                if (e.sensor.type == Sensor.TYPE_ACCELEROMETER) {
                    for (i in 0..2) accSum[i] += e.values[i]
                    counts[0]++
                } else if (e.sensor.type == Sensor.TYPE_GYROSCOPE) {
                    for (i in 0..2) gyroSum[i] += e.values[i]
                    counts[1]++
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        if (acc == null || gyro == null) sensorMissing = true
        else {
            sm.registerListener(listener, acc, 50_000)
            sm.registerListener(listener, gyro, 50_000)
        }
        onDispose { sm.unregisterListener(listener) }
    }

    // 2) built-in step counter (reference only)
    DisposableEffect(stepPerm) {
        val sm = context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
        val stepSensor = sm.getDefaultSensor(Sensor.TYPE_STEP_COUNTER)
        var first = true
        val l = object : SensorEventListener {
            override fun onSensorChanged(e: SensorEvent) {
                val total = e.values[0]
                val last = prefs.getFloat("last_total", -1f)
                prefs.edit().putFloat("last_total", total).apply()
                if (first) { first = false; return }
                if (last < 0f || total < last) return
                val delta = (total - last).toInt()
                if (delta <= 0) return
                val d = LocalDate.now().toString()
                val raw = prefs.getInt("raw_$d", 0) + delta
                prefs.edit().putInt("raw_$d", raw).apply()
                builtInSteps = raw
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        if (stepSensor == null) noStepSensor = true
        else if (stepPerm) sm.registerListener(l, stepSensor, SensorManager.SENSOR_DELAY_NORMAL)
        onDispose { sm.unregisterListener(l) }
    }

    // 3) every 50 ms: 20 Hz row, step detection; every second: predict, smooth, log, reminder
    LaunchedEffect(interpreter) {
        var next = SystemClock.elapsedRealtime()
        var tickCount = 0
        var rowCounter = 0
        var secStart = next
        while (true) {
            next += 50
            val wait = next - SystemClock.elapsedRealtime()
            if (wait > 0) delay(wait)
            else if (wait < -200) next = SystemClock.elapsedRealtime()

            if (counts[0] > 0) { for (i in 0..2) lastAcc[i] = accSum[i] / counts[0]; seen[0] = true }
            if (counts[1] > 0) { for (i in 0..2) lastGyro[i] = gyroSum[i] / counts[1]; seen[1] = true }
            for (i in 0..2) { accSum[i] = 0f; gyroSum[i] = 0f }
            counts[0] = 0; counts[1] = 0
            if (!(seen[0] && seen[1])) continue

            val row = floatArrayOf(lastAcc[0], lastAcc[1], lastAcc[2], lastGyro[0], lastGyro[1], lastGyro[2])
            if (phase == 2) recRows.add(row)

            // step detection (v1): bumps in the acceleration magnitude
            val mag = sqrt(row[0] * row[0] + row[1] * row[1] + row[2] * row[2])
            base[0] = 0.95f * base[0] + 0.05f * mag
            val dev = mag - base[0]
            val nowMs = SystemClock.elapsedRealtime()
            if (dev1[0] > dev2[0] && dev1[0] >= dev && dev1[0] > STEP_THRESHOLD && nowMs - lastStepMs[0] > MIN_STEP_MS) {
                lastStepMs[0] = nowMs
                if (moving[0]) commit(1) else { pend[0] += 1; pendT[0] = nowMs }
            }
            dev2[0] = dev1[0]
            dev1[0] = dev

            rows.addLast(row)
            if (rows.size > WINDOW) rows.removeFirst()
            filled = rows.size
            rowCounter++
            tickCount++

            if (SystemClock.elapsedRealtime() - secStart >= 1000) {
                rowsPerSec = rowCounter
                rowCounter = 0
                secStart = SystemClock.elapsedRealtime()
            }

            if (interpreter != null && rows.size == WINDOW && tickCount % 20 == 0) {
                val input = Array(1) { Array(WINDOW) { FloatArray(N_CH) } }
                for (t in 0 until WINDOW) {
                    for (c in 0 until N_CH) input[0][t][c] = (rows[t][c] - MEAN[c]) / STD[c]
                }
                val output = Array(1) { FloatArray(4) }
                interpreter.run(input, output)
                probs = output[0].clone()
                val best = probs.indices.maxByOrNull { probs[it] } ?: 0
                rawLabel = LABELS[best]

                recent.addLast(best)
                if (recent.size > 5) recent.removeFirst()
                val votes = IntArray(4)
                for (r in recent) votes[r]++
                var top = recent.last()
                for (c in 0..3) if (votes[c] > votes[top]) top = c
                smoothLabel = LABELS[top]

                moving[0] = (top == 0 || top == 1)
                if (pend[0] > 0) {
                    if (moving[0]) { commit(pend[0]); pend[0] = 0 }
                    else if (SystemClock.elapsedRealtime() - pendT[0] > 8000) { ignore(pend[0]); pend[0] = 0 }
                }

                val day = LocalDate.now().toString()
                val key = "t_${day}_${LABELS[top]}"
                prefs.edit().putLong(key, prefs.getLong(key, 0L) + 1L).apply()
                todaySecs = LongArray(4) { prefs.getLong("t_${day}_${LABELS[it]}", 0L) }

                // sitting reminder: how long has the person been still?
                if (top == 2 || top == 3) {
                    stillSecs++
                    if (!reminded[0] && stillSecs >= reminderMin * 60) {
                        reminded[0] = true
                        sendReminder(context, "You have been still for ${stillSecs / 60} min. Time for a short walk!")
                    }
                } else { stillSecs = 0; reminded[0] = false }
            }
        }
    }

    // 4) recording timer
    LaunchedEffect(startToken) {
        if (startToken == 0) return@LaunchedEffect
        message = ""
        phase = 1
        for (s in PREP_SECONDS downTo 1) { secondsLeft = s; delay(1000) }
        recRows.clear()
        tone.startTone(ToneGenerator.TONE_PROP_BEEP, 300)
        phase = 2
        for (s in RECORD_SECONDS downTo 1) { secondsLeft = s; delay(1000) }
        phase = 0
        tone.startTone(ToneGenerator.TONE_PROP_BEEP2, 700)
        val name = LABELS[selected]
        val session = System.currentTimeMillis()
        val copy = ArrayList(recRows)
        val sb = StringBuilder("ax,ay,az,gx,gy,gz,label,session\n")
        for (r in copy) sb.append(r.joinToString(",")).append(",").append(name).append(",").append(session).append("\n")
        val where = saveCsv(context, "own_${name.lowercase()}_${POSITION}_$session.csv", sb.toString())
        message = "Saved ${copy.size} rows: $where"
    }

    DisposableEffect(interpreter) { onDispose { interpreter?.close(); tone.release() } }

    // ---------------- screen ----------------
    val activeMin = (todaySecs[0] + todaySecs[1]) / 60
    val actIndex = LABELS.indexOf(smoothLabel)
    val actColor = ACT_COLORS.getOrElse(actIndex) { Grey }

    Column(modifier.fillMaxSize().background(Bg)) {
        Text("Activity Tracker", fontSize = 24.sp, fontWeight = FontWeight.Bold, color = Ink,
            modifier = Modifier.padding(start = 16.dp, top = 12.dp))
        Row(Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 8.dp), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            tabs.forEachIndexed { i, t ->
                if (tab == i) Button(onClick = { tab = i }, modifier = Modifier.weight(1f), contentPadding = PaddingValues(horizontal = 4.dp)) { Text(t, fontSize = 13.sp) }
                else OutlinedButton(onClick = { tab = i }, modifier = Modifier.weight(1f), contentPadding = PaddingValues(horizontal = 4.dp)) { Text(t, fontSize = 13.sp) }
            }
        }

        Column(
            Modifier.weight(1f).verticalScroll(rememberScrollState()).padding(horizontal = 16.dp).padding(bottom = 16.dp),
            verticalArrangement = Arrangement.spacedBy(12.dp)
        ) {
            if (sensorMissing) Text("A required sensor is missing on this phone!", color = Color.Red)
            if (loadError != null) Text("Model load failed: $loadError", color = Color.Red)

            // ===== TODAY =====
            if (tab == 0) {
                val d = LocalDate.now()
                Text("${d.dayOfWeek.getDisplayName(TextStyle.FULL, Locale.getDefault())}, ${d.dayOfMonth} ${d.month.getDisplayName(TextStyle.SHORT, Locale.getDefault())}", color = Grey)

                AppCard {
                    Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                        Ring(ownSteps / goalSteps.toFloat(), Teal, Modifier.size(150.dp)) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Text("$ownSteps", fontSize = 30.sp, fontWeight = FontWeight.Bold, color = Ink)
                                Text("steps", color = Grey, fontSize = 13.sp)
                            }
                        }
                        Column(verticalArrangement = Arrangement.spacedBy(6.dp)) {
                            Text("Goal: $goalSteps steps", color = Ink)
                            Text("Active: $activeMin / $goalActive min", color = Ink)
                            Text("${(100 * ownSteps / goalSteps.toFloat()).toInt().coerceAtMost(999)}% of today's goal", color = Grey, fontSize = 13.sp)
                        }
                    }
                }

                AppCard {
                    Text("Right now", color = Grey, fontSize = 13.sp)
                    if (filled < WINDOW) {
                        Text("Warming up…", fontSize = 26.sp, fontWeight = FontWeight.Bold, color = Grey)
                        Bar(filled / WINDOW.toFloat(), Grey)
                    } else {
                        Text(smoothLabel, fontSize = 30.sp, fontWeight = FontWeight.Bold, color = actColor)
                        Text("Latest single prediction: $rawLabel", color = Grey, fontSize = 13.sp)
                    }
                }

                AppCard {
                    Text("Time today", color = Grey, fontSize = 13.sp)
                    val maxSecs = maxOf(1L, todaySecs.maxOrNull() ?: 1L)
                    LABELS.forEachIndexed { i, name ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(name, color = Ink)
                            Text(fmt(todaySecs[i]), color = Grey)
                        }
                        Bar(todaySecs[i] / maxSecs.toFloat(), ACT_COLORS[i])
                    }
                }

                AppCard {
                    Text("Sitting reminder", color = Grey, fontSize = 13.sp)
                    Text("Still for ${stillSecs / 60} min (reminder at $reminderMin min)", color = Ink)
                    Bar(stillSecs / (reminderMin * 60f), Orange)
                }
            }

            // ===== WEEK =====
            if (tab == 1) {
                val days = (6 downTo 0).map { LocalDate.now().minusDays(it.toLong()) }
                val dayLabels = days.map { it.dayOfWeek.getDisplayName(TextStyle.SHORT, Locale.getDefault()) }
                val weekSteps = days.map { prefs.getInt("own_$it", 0).toFloat() }
                val weekActive = days.map { (prefs.getLong("t_${it}_Walking", 0L) + prefs.getLong("t_${it}_Jogging", 0L)) / 60f }
                val bestIdx = weekSteps.indices.maxByOrNull { weekSteps[it] } ?: 0

                AppCard {
                    Text("Steps, last 7 days", fontWeight = FontWeight.Bold, color = Ink)
                    BarChart(weekSteps, dayLabels, Teal, goalSteps.toFloat())
                    Text("Total: ${weekSteps.sum().toInt()}   Daily average: ${(weekSteps.sum() / 7).toInt()}", color = Ink)
                    Text("Best day: ${dayLabels[bestIdx]} (${weekSteps[bestIdx].toInt()} steps). Today so far: $ownSteps", color = Grey, fontSize = 13.sp)
                }
                AppCard {
                    Text("Active minutes (walking + jogging)", fontWeight = FontWeight.Bold, color = Ink)
                    BarChart(weekActive, dayLabels, Orange, goalActive.toFloat())
                    Text("Total: ${weekActive.sum().toInt()} min   Daily average: ${(weekActive.sum() / 7).toInt()} min", color = Ink)
                }
                Text("Bars in full colour reached the daily goal. The app counts only while it is open.", color = Grey, fontSize = 12.sp)
            }

            // ===== GOALS =====
            if (tab == 2) {
                AppCard {
                    Text("Daily steps goal", fontWeight = FontWeight.Bold, color = Ink)
                    Text("$goalSteps steps", fontSize = 22.sp, color = Teal, fontWeight = FontWeight.Bold)
                    Bar(ownSteps / goalSteps.toFloat(), Teal)
                    Text("Today: $ownSteps", color = Grey, fontSize = 13.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = { goalSteps = maxOf(1000, goalSteps - 500); prefs.edit().putInt("goal_steps", goalSteps).apply() }) { Text("− 500") }
                        OutlinedButton(onClick = { goalSteps += 500; prefs.edit().putInt("goal_steps", goalSteps).apply() }) { Text("+ 500") }
                    }
                }
                AppCard {
                    Text("Daily active minutes goal", fontWeight = FontWeight.Bold, color = Ink)
                    Text("$goalActive min", fontSize = 22.sp, color = Orange, fontWeight = FontWeight.Bold)
                    Bar(activeMin / goalActive.toFloat(), Orange)
                    Text("Today: $activeMin min", color = Grey, fontSize = 13.sp)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = { goalActive = maxOf(5, goalActive - 5); prefs.edit().putInt("goal_active", goalActive).apply() }) { Text("− 5") }
                        OutlinedButton(onClick = { goalActive += 5; prefs.edit().putInt("goal_active", goalActive).apply() }) { Text("+ 5") }
                    }
                }
                AppCard {
                    Text("Sitting reminder", fontWeight = FontWeight.Bold, color = Ink)
                    Text("Notify me after $reminderMin min without moving", color = Ink)
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        OutlinedButton(onClick = {
                            val i = REMINDER_OPTIONS.indexOf(reminderMin).coerceAtLeast(0)
                            reminderMin = REMINDER_OPTIONS[maxOf(0, i - 1)]; prefs.edit().putInt("reminder_min", reminderMin).apply()
                        }) { Text("Shorter") }
                        OutlinedButton(onClick = {
                            val i = REMINDER_OPTIONS.indexOf(reminderMin).coerceAtLeast(0)
                            reminderMin = REMINDER_OPTIONS[minOf(REMINDER_OPTIONS.size - 1, i + 1)]; prefs.edit().putInt("reminder_min", reminderMin).apply()
                        }) { Text("Longer") }
                    }
                    Button(onClick = { sendReminder(context, "This is a test reminder. Time for a short walk!") }) { Text("Send test notification") }
                    Text("Reminders work while the app is open with the screen on.", color = Grey, fontSize = 12.sp)
                }
            }

            // ===== DATA =====
            if (tab == 3) {
                AppCard {
                    Text("Step counters", fontWeight = FontWeight.Bold, color = Ink)
                    Text("Our detector: $ownSteps  (rejected: $ownIgnored)", color = Ink)
                    Text(
                        when {
                            noStepSensor -> "Built-in sensor: this phone has none"
                            !stepPerm -> "Built-in sensor: permission not granted"
                            else -> "Built-in sensor: $builtInSteps"
                        }, color = Grey
                    )
                    Button(onClick = { resetSteps() }) { Text("Reset step counters") }
                }
                AppCard {
                    Text("Model output", fontWeight = FontWeight.Bold, color = Ink)
                    LABELS.forEachIndexed { i, name ->
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(name, color = Ink)
                            Text("${"%.0f".format(probs[i] * 100)}%", color = Grey)
                        }
                        Bar(probs[i], ACT_COLORS[i])
                    }
                    Text("Rows per second (about 20): $rowsPerSec   Barometer: ${if (hasBarometer) "yes" else "no"}", color = Grey, fontSize = 12.sp)
                }
                AppCard {
                    Text("Record your own data", fontWeight = FontWeight.Bold, color = Ink)
                    Text("Pick an activity, press Start, then put the phone in your $POSITION within $PREP_SECONDS seconds. It beeps to start and stop.", color = Grey, fontSize = 13.sp)
                    LABELS.forEachIndexed { i, name ->
                        if (selected == i) Button(onClick = { selected = i }, enabled = phase == 0) { Text("✓ $name") }
                        else OutlinedButton(onClick = { selected = i }, enabled = phase == 0) { Text(name) }
                    }
                    Button(onClick = { startToken++ }, enabled = phase == 0) { Text("Start recording ${LABELS[selected]}") }
                    Text(
                        when (phase) {
                            1 -> "Get ready... $secondsLeft"
                            2 -> "RECORDING ${LABELS[selected]}: $secondsLeft s left"
                            else -> "Idle"
                        }, color = Ink
                    )
                    if (message.isNotEmpty()) Text(message, color = Grey, fontSize = 12.sp)
                }
            }
        }
    }
}