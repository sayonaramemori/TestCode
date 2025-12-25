<template>
  <!-- <ThreeScene /> -->
    <div id="wrapper">
        <div id="d2">
  <ParticleSys />
        </div>
        <div id="d3">
  <ParticleSys3D />
        </div>
    </div>
</template>

<script setup>
import ThreeScene from './components/ThreeScene.vue'
import ParticleSys from './components/ParticleSys.vue'
import ParticleSys3D from './components/ParticleSys3D.vue'

</script>

<style scoped>
#wrapper {
    width: 1200px;
}
#d2 {
    width: 600px;
    height: 600px;
    background-color: black;
    display: inline-block;
}
#d3 {
    width: 600px;
    height: 600px;
    background-color: white;
    display: inline-block;
}
.logo {
  height: 6em;
  padding: 1.5em;
  will-change: filter;
  transition: filter 300ms;
}
.logo:hover {
  filter: drop-shadow(0 0 2em #646cffaa);
}
.logo.vue:hover {
  filter: drop-shadow(0 0 2em #42b883aa);
}
</style>
