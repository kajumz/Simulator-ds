.PHONY: seo seo-stop

LOGFILE := streamlit.log

seo:
	@echo "Starting the Streamlit App..."
	@streamlit run app.py > $(LOGFILE) 2>&1 &

seo-stop:
	@echo "Stopping the Streamlit App..."
	@pkill -f "streamlit run app.py"
