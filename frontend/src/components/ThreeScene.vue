<template>
  <div ref="container" class="three-container"></div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, ref } from 'vue'
import * as THREE from 'three'

const container = ref(null)

let scene, camera, renderer
let cube
let animationId

function initScene() {
  // Scene
  scene = new THREE.Scene()
  scene.background = new THREE.Color(0x000000)

  // Camera
  const { clientWidth: w, clientHeight: h } = container.value
  camera = new THREE.PerspectiveCamera(60, w / h, 0.1, 1000)
  camera.position.set(3, 3, 3)
  camera.lookAt(0, 0, 0)

  // Renderer
  renderer = new THREE.WebGLRenderer({ antialias: true })
  renderer.setSize(w, h)
  renderer.setPixelRatio(window.devicePixelRatio)
  container.value.appendChild(renderer.domElement)

  // Light
  const light = new THREE.DirectionalLight(0xffffff, 1)
  light.position.set(0, 5, 5)
  scene.add(light)

  const ambient = new THREE.AmbientLight(0x404040)
  scene.add(ambient)

  // Geometry
  const geometry = new THREE.BoxGeometry()
  const material = new THREE.MeshStandardMaterial({
    color: 0x4ade80
  })

  cube = new THREE.Mesh(geometry, material)
  scene.add(cube)

  // Resize
  window.addEventListener('resize', onResize)
}

function animate() {
  cube.rotation.x += 0.005
  cube.rotation.y += 0.005

  renderer.render(scene, camera)
  animationId = requestAnimationFrame(animate)
}

function onResize() {
  const { clientWidth: w, clientHeight: h } = container.value
  camera.aspect = w / h
  camera.updateProjectionMatrix()
  renderer.setSize(w, h)
}

onMounted(() => {
  initScene()
  animate()
})

onBeforeUnmount(() => {
  cancelAnimationFrame(animationId)
  window.removeEventListener('resize', onResize)
  renderer.dispose()
})
</script>

<style scoped>
.three-container {
  width: 100%;
  min-width: 1000px;
  height: 100vh;
  overflow: hidden;
}
</style>

