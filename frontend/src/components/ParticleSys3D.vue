<template>
  <div ref="container" class="canvas"></div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue'
import * as THREE from 'three'

let field = null
const loading = ref(true)
const error = ref(null)

const container = ref(null)

let scene, camera, renderer
let particles, geometry
let positions
let animationId

const N = 32000
const dt = 0.081
const xbounds = 0.8
const ybounds = 1.0

function addBoundary() {
  const w = 0.5
  const h = 0.5
  const points = [
    new THREE.Vector3(0, 0, 0),
    new THREE.Vector3(0.8,0, 0),
    new THREE.Vector3(0.8,1, 0),
    new THREE.Vector3(0, 1, 0),
  ]
  const geometry = new THREE.BufferGeometry().setFromPoints(points)

  const material = new THREE.LineBasicMaterial({
    color: 0xffffff
  })

  const boundary = new THREE.LineLoop(geometry, material)
  scene.add(boundary)
}

/* -----------------------------
   Velocity field u(x, y)
------------------------------ */
function velocity(x, y) {
  const r2 = x * x + y * y
  return {
    u: -y/(r2+0.3),
    v: x/(r2+0.3)
  }
}

/* -----------------------------
   Initialize scene
------------------------------ */
function init() {
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  camera = new THREE.OrthographicCamera(
    -0.5, 0.5, 0.5, -0.5, 0.1, 1000
  )
  camera.position.z = 2
  camera.position.x = 0.4
  camera.position.y = 0.5

  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(
    container.value.clientWidth,
    container.value.clientHeight
  )
  renderer.setPixelRatio(window.devicePixelRatio)
  container.value.appendChild(renderer.domElement)

  geometry = new THREE.BufferGeometry()
  positions = new Float32Array(N * 3)  //xy

  for (let i = 0; i < N; i++) {
    // positions[3 * i]     = (Math.random() * 2 - 1)
    // positions[3 * i + 1] = (Math.random() * 2 - 1)
    positions[3 * i]     =  Math.random() * 0.8
    positions[3 * i + 1] =  Math.random()
    positions[3 * i + 2] = 0
  }

  geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(positions, 3)
  )

  const material = new THREE.PointsMaterial({
    color: 0x38bdf8,
    size: 1.5
  })

  particles = new THREE.Points(geometry, material)
  scene.add(particles)
  addBoundary()

  window.addEventListener('resize', resize)
}

/* -----------------------------
   Time integration
------------------------------ */
function updateParticles() {
  for (let i = 0; i < N; i++) {
    let x = positions[3 * i]
    let y = positions[3 * i + 1]

    if(loading.value){
      const { u, v } = velocity(x, y)
      x += u * dt
      y += v * dt
    }else{
      const { u, v } = velocityAt(x, y)
      x += u * dt
      y += v * dt
    }

    // Re-inject particles
    // if (abs(x) > bounds || Math.abs(y) > bounds) {
    if (x > xbounds || y > ybounds || x < 0 || y <0) {
      x = Math.random() * 0.8
      y = Math.random()
    }

    positions[3 * i]     = x
    positions[3 * i + 1] = y
  }

  geometry.attributes.position.needsUpdate = true
}

/* -----------------------------
   Animation loop
------------------------------ */
function animate() {
  updateParticles()
  renderer.render(scene, camera)
  animationId = requestAnimationFrame(animate)
}

/* -----------------------------
   Resize
------------------------------ */
function resize() {
  const w = container.value.clientWidth
  const h = container.value.clientHeight
  renderer.setSize(w, h)
}


async function loadVelocityField() {
    // const res = await fetch('http://localhost:8000/vel')
    const res = await fetch('http://localhost:8000/vel3d')
    const json = await res.json()
    const data = json.Data
  // console.log(json.Data)
    return {
      xmin: data.xmin,
      xmax: data.xmax,
      ymin: data.ymin,
      ymax: data.ymax,
      nx: data.nx,
      ny: data.ny,
      u: new Float32Array(data.u),
      v: new Float32Array(data.v)
    }
}

function velocityAt(x, y) {
  let xmi = field.xmin
  let xma = field.xmax
  let ymi = field.ymin
  let yma = field.ymax
  let nx = field.nx 
  let ny = field.ny
  if ( x < xmi || x > xma || y < ymi || y > yma) {
    return { u: 1, v: 1 }
  }
  const fx = (x - xmi) / (xma - xmi) * (nx - 1)
  const fy = (y - ymi) / (yma - ymi) * (ny - 1)

  const i = Math.floor(fx)
  const j = Math.floor(fy)
  // Prevent out-of-bounds access
  if (i < 0 || i >= nx - 1 || j < 0 || j >= ny - 1) {
    return { u: 1, v: 1 }
  }
  const tx = fx - i
  const ty = fy - j

  const idx = (ii, jj) => ii + jj * nx

  const u00 = field.u[idx(i,   j  )]
  const u10 = field.u[idx(i+1, j  )]
  const u01 = field.u[idx(i,   j+1)]
  const u11 = field.u[idx(i+1, j+1)]

  const v00 = field.v[idx(i,   j  )]
  const v10 = field.v[idx(i+1, j  )]
  const v01 = field.v[idx(i,   j+1)]
  const v11 = field.v[idx(i+1, j+1)]

  const uu =
    (1-tx)*(1-ty)*u00 +
    tx*(1-ty)*u10 +
    (1-tx)*ty*u01 +
    tx*ty*u11

  const vv =
    (1-tx)*(1-ty)*v00 +
    tx*(1-ty)*v10 +
    (1-tx)*ty*v01 +
    tx*ty*v11

  if(uu==0 && vv ==0){
    return { u: 0.01, v: 0.01}
  }
  return { u: uu, v: vv }
}

/* -----------------------------
   Lifecycle
------------------------------ */
onMounted(async () => {
  field = await loadVelocityField()
  if(field.nx != undefined){
    loading.value = false
  }else{
    console.log("failed")
  }
  init()
  animate()
})

onBeforeUnmount(() => {
  cancelAnimationFrame(animationId)
  window.removeEventListener('resize', resize)
  renderer.dispose()
})
</script>

<style scoped>
.canvas {
  /* width: 100%; */
  /* min-width: 500px; */
  width: 600px;
  height: 600px;
  border-radius: 10px;
  border-width: 10px;
  /* height: 100vh; */
}
</style>

