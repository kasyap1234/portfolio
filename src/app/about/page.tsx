export default function About() {
  return (
    <div className="min-h-screen bg-gradient-to-b from-gray-50 to-white dark:from-gray-900 dark:to-gray-800">
      <div className="max-w-4xl mx-auto px-6 py-16 space-y-12 animate-fade-in-down">
        <h1 className="text-5xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-gray-950 to-gray-650 text-center mb-12 animate-slide-in-top">
          About Me
        </h1>
        
        <section className="space-y-12">
          {/* Each card gets staggered animation */}
          <div className="bg-white shadow-xl  dark:bg-gray-800 rounded-xl  p-8 hover:shadow-2xl transition-all duration-300 hover:scale-[1.02] animate-slide-in-left">
            <h2 className="text-3xl font-semibold text-gray-950 dark:text-blue-400 mb-4">Introduction</h2>
            <p className="text-lg leading-relaxed text-gray-700 dark:text-gray-300">
              Hi, I'm Kasyap Dharanikota, a passionate software developer with a strong focus on backend systems 
              and full-stack development. My expertise spans <span className="font-medium text-blue-600 dark:text-blue-400 hover:text-teal-500 transition-colors">Golang</span>, {' '}
              <span className="font-medium text-blue-600 dark:text-blue-400 hover:text-teal-500 transition-colors">NestJS</span>, {' '}
              <span className="font-medium text-blue-600 dark:text-blue-400 hover:text-teal-500 transition-colors">JavaScript</span>, {' '}
              <span className="font-medium text-blue-600 dark:text-blue-400 hover:text-teal-500 transition-colors">TypeScript</span>, as well as {' '}
              <span className="font-medium text-blue-600 dark:text-blue-400 hover:text-teal-500 transition-colors">ReactJS</span> and {' '}
              <span className="font-medium text-blue-600 dark:text-blue-400 hover:text-teal-500 transition-colors">TailwindCSS</span> for frontend development.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 hover:shadow-xl transition-all duration-300 hover:scale-[1.02] animate-slide-in-right">
            <h2 className="text-3xl font-semibold text-gray-950 dark:text-blue-400 mb-4">What I Do</h2>
            <p className="text-lg leading-relaxed text-gray-700 dark:text-gray-300">
              I specialize in developing robust APIs, microservices, and SaaS products with a focus on 
              scalability, efficiency, and reliability. My backend experience is complemented by a solid 
              foundation in frontend development, where I create responsive, user-friendly interfaces.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 hover:shadow-xl transition-all duration-300 hover:scale-[1.02] animate-slide-in-left">
            <h2 className="text-3xl font-semibold text-gray-950 dark:text-blue-400 mb-4">Current Projects</h2>
            <p className="text-lg leading-relaxed text-gray-700 dark:text-gray-300">
              Currently learning advanced backend concepts and cloud native technologies.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 hover:shadow-xl transition-all duration-300 hover:scale-[1.02] animate-slide-in-right">
            <h2 className="text-3xl font-semibold text-gray-950 dark:text-blue-400 mb-4">Cloud & DevOps</h2>
            <p className="text-lg leading-relaxed text-gray-700 dark:text-gray-300">
              Experienced in deploying applications using Kubernetes, Docker containers, and AWS cloud 
              services. Passionate about modern DevOps practices and cloud-native solutions.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 hover:shadow-xl transition-all duration-300 hover:scale-[1.02] animate-slide-in-left">
            <h2 className="text-3xl font-semibold text-gray-950 dark:text-blue-400 mb-4">Beyond Coding</h2>
            <p className="text-lg leading-relaxed text-gray-700 dark:text-gray-300">
              Enthusiastic about personal finance and investing, maintaining a diverse portfolio across 
              financial services, technology, healthcare, and consumer goods sectors.
            </p>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-lg p-8 hover:shadow-xl transition-all duration-300 hover:scale-[1.02] animate-slide-in-right">
            <h2 className="text-3xl font-semibold text-gray-950 dark:text-blue-400 mb-4">Let's Connect</h2>
            <p className="text-lg leading-relaxed text-gray-700 dark:text-gray-300">
              Interested in collaboration or discussion? Feel free to reach out! Always open to new 
              projects and innovative solutions.
            </p>
          </div>
        </section>
      </div>
    </div>
  )
}
